"""Semantic-search agent smoke test (no Redis / no network).

Asserts: natural-language product queries surface ``semantic_search`` against the
vector store; the bounded agent loop feeds the KB result back so the model then
calls ``get_product_detail(product_id=...)`` (a combination of tools for one
request); real tool schemas are advertised; and no token ever leaks into the
emitted events.
"""
import json
import os

from gateway.agent import Agent
from gateway.debug import DebugTracer
from gateway.session import Session
from mcp_server.guards import ConfirmationGate
from mcp_server.tools_catalog import parameters_for

HERE = os.path.dirname(__file__)
SYSTEM_PROMPT = open(os.path.join(HERE, "..", "..", "prompts", "system.md"), encoding="utf-8").read()


class FakeModel:
    def __init__(self, script):
        self._script, self.calls, self.received = script, 0, []

    @property
    def provider_name(self):
        return "stub"

    @property
    def model_name(self):
        return "stub-model"

    async def stream(self, messages, tools, system_prompt, temperature=0.7, max_tokens=4096):
        self.received.append(messages)
        events = self._script[self.calls] if self.calls < len(self._script) else [{"type": "finish", "reason": "end_turn"}]
        self.calls += 1
        for e in events:
            yield e

    async def classify(self, message, system_prompt, valid_intents, temperature=0.0):
        return {"intent": "info", "confidence": 0.99, "reasoning": "smoke", "error": None}


class FakeVectorStore:
    def __init__(self, items):
        self._items, self.queries = items, []

    async def search(self, query, top_k=5):
        self.queries.append(query)
        return self._items[: max(1, min(top_k, 20))]


class FakeBackend:
    def __init__(self):
        self.calls, self.last_pp = 0, {}

    async def call(self, method, path, *, token=None, params=None, path_params=None,
                   json_body=None, idempotency_key=None, correlation_id=None):
        self.calls += 1
        self.last_pp = path_params or {}
        if path_params and path_params.get("product_id") == "prod-50":
            return {"id": "prod-50", "title": "Red Leather Wallet", "merchant_id": "m-1",
                    "merchant_name": "Leather Co", "category_id": "cat-9", "category_name": "Accessories",
                    "base_price_amount": 4800.0, "base_price_currency": "NGN", "status": "active",
                    "image_url": "https://example.com/wallet.jpg", "quantity_available": 3,
                    "description": "Hand-stitched red leather wallet.",
                    "variants": [{"id": "v-1", "sku": "RLW-BLK", "name": "Black",
                                  "base_price_amount": 4800.0, "base_price_currency": "NGN",
                                  "quantity_available": 3, "attributes": {}}],
                    "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z"}
        return {"items": []}


def _agent():
    doc = {"id": "doc-1", "text": "Hand-stitched red leather wallet, about $48. product_id=prod-50",
           "metadata": {"product_id": "prod-50", "source": "product"}, "score": 0.95}
    vs = FakeVectorStore([doc])
    backend = FakeBackend()
    model = FakeModel([
        [{"type": "tool_calls", "tool_calls": [
            {"id": "tc-1", "name": "semantic_search", "arguments": {"query": "red leather wallet", "top_k": 5}}]},
         {"type": "finish", "reason": "tool_use"}],
        [{"type": "tool_calls", "tool_calls": [
            {"id": "tc-2", "name": "get_product_detail", "arguments": {"product_id": "prod-50"}}]}],
        [{"type": "delta", "content": "Here is the red leather wallet you asked for:"},
         {"type": "finish", "reason": "end_turn"}],
    ])
    return (Agent(backend_client=backend, gate=ConfirmationGate(redis_client=None, namespace="agent"),
                  model_backend=model,
                  debug_tracer=DebugTracer(redis_client=None, namespace="agent", enabled=False),
                  system_prompt=SYSTEM_PROMPT, vector_store=vs, vector_namespace="agent"),
            vs, backend, model)


async def test_natural_language_query_runs_semantic_search_then_detail():
    agent, vs, backend, model = _agent()
    session = Session(session_id="smoke-1", role=None)
    events = [ev async for ev in agent.run_turn(session, "I'm looking for a red leather wallet")]

    assert "query" in parameters_for("semantic_search")["properties"]
    assert parameters_for("semantic_search")["required"] == ["query"]
    assert "q" in parameters_for("search_products")["properties"]
    assert vs.queries and "red leather wallet" in vs.queries[0]
    assert backend.last_pp.get("product_id") == "prod-50"
    assert backend.calls == 1
    assert model.calls == 3
    assert "tool" in [m.get("role") for m in model.received[1]]

    directives = [e["directive"] for e in events if e.get("type") == "ui_directive"]
    assert any(d["component"] == "ProductGrid" for d in directives)
    detail = next(d for d in directives if d["component"] == "ProductDetail")
    assert detail["props"]["name"] == "Red Leather Wallet"
    grid = next(d for d in directives if d["component"] == "ProductGrid")
    assert grid["props"]["items"][0]["text"] == vs._items[0]["text"]
    blob = json.dumps(events).lower()
    assert "access_token" not in blob and "user_token" not in blob


async def test_semantic_search_without_vector_store_degrades_gracefully():
    agent, _, _, _ = _agent()
    agent._vector_store = None
    agent._redis = None
    session = Session(session_id="smoke-grace", role=None)
    events = [ev async for ev in agent.run_turn(session, "anything")]
    texts = [e.get("content") for e in events if e.get("type") == "text"]
    # A semantic_search miss must not be branded as a "knowledge base" lookup —
    # product queries route here, so report it as no results and let the turn
    # fall back to category browsing instead of mentioning the knowledge base.
    assert any("couldn't find any results" in t for t in texts)
    assert not any("knowledge base" in t.lower() for t in texts)


async def test_handles_openai_nested_tool_call_format():
    """Regression guard: the live NVIDIA backend emits OpenAI nested
    {function:{name, arguments(str)}}; the agent must normalize that shape
    (this was the '[None]' / 'Unknown tool' bug)."""
    agent, vs, backend, model = _agent()
    session = Session(session_id="smoke-openai", role=None)
    args = json.dumps({"query": "red leather wallet", "top_k": 5})
    event = {"type": "tool_calls", "tool_calls": [
        {"id": "tc-x", "type": "function", "function": {"name": "semantic_search", "arguments": args}}
    ]}
    outputs = [o async for o in agent._handle_event(event, session, 1, None, [])]
    assert vs.queries and "red leather wallet" in vs.queries[0]
    dirs = [o["directive"] for o in outputs if o.get("type") == "ui_directive"]
    assert any(d["component"] == "ProductGrid" for d in dirs)



async def test_repeated_identical_tool_call_does_not_loop():
    """Regression guard for the live 'too many steps' loop: re-proposing the SAME
    tool call (identical name + args) must be skipped, so the turn terminates with
    the already-fetched results instead of spinning against the iteration cap."""
    agent, vs, backend, model = _agent()
    dup = {"query": "phone case", "top_k": 5}
    # Round 1: semantic_search + finish(tool_use). Round 2: the SAME semantic_search
    # call again (a degenerate re-issue) with no finish -> must be deduped and skipped.
    model._script = [
        [{"type": "tool_calls", "tool_calls": [{"id": "a", "name": "semantic_search", "arguments": dup}]},
         {"type": "finish", "reason": "tool_use"}],
        [{"type": "tool_calls", "tool_calls": [{"id": "b", "name": "semantic_search", "arguments": dup}]}],
    ]
    session = Session(session_id="smoke-dedup", role=None)
    events = [ev async for ev in agent.run_turn(session, "phone case")]
    errors = [e for e in events if e.get("type") == "error"]
    assert not any("too many steps" in str(e) for e in errors), "degenerate loop was not stopped"
    assert vs.queries.count("phone case") == 1, "duplicate call was re-executed"
    dirs = [e["directive"] for e in events if e.get("type") == "ui_directive"]
    assert any(d["component"] == "ProductGrid" for d in dirs), "round-1 results were not returned to the UI"
    assert model.calls == 2


class _ProductBackend(FakeBackend):
    """FakeBackend that can surface catalog products for /catalog/products."""

    def __init__(self, products=None):
        super().__init__()
        self.products = products or []

    async def call(self, method, path, *, token=None, params=None, path_params=None,
                   json_body=None, idempotency_key=None, correlation_id=None):
        self.calls += 1
        self.last_pp = path_params or {}
        if path == "/catalog/products":
            return {"items": self.products}
        if path_params and path_params.get("product_id") == "prod-50":
            return {"id": "prod-50", "title": "Red Leather Wallet", "merchant_id": "m-1",
                    "merchant_name": "Leather Co", "category_id": "cat-9",
                    "base_price_amount": 4800.0, "base_price_currency": "NGN",
                    "status": "active", "image_url": "https://example.com/wallet.jpg",
                    "quantity_available": 3}
        return {"items": []}


def _agent_for(products=None, vector_items=None, script=None):
    backend = _ProductBackend(products=products or [])
    vs = FakeVectorStore(vector_items or [])
    model = FakeModel(script or [])
    agent = Agent(backend_client=backend, gate=ConfirmationGate(redis_client=None, namespace="agent"),
                  model_backend=model,
                  debug_tracer=DebugTracer(redis_client=None, namespace="agent", enabled=False),
                                    system_prompt=SYSTEM_PROMPT, vector_store=vs, vector_namespace="agent")
    return agent, vs, backend, model


async def test_semantic_miss_suppressed_when_product_search_hits():
    """Regression for the 'couldn't find any results ... Found N products'
    contradiction: when semantic_search returns nothing but search_products
    returns items in the same stream, the semantic_search miss summary must
    NOT be emitted — the transcript should only show the hit.
    """
    prod = {"id": "prod-50", "title": "Power Bank 20000mAh", "merchant_id": "m-1",
            "merchant_name": "Gadget Store", "category_id": "cat-3",
            "base_price_amount": 12000.0, "base_price_currency": "NGN",
            "status": "active", "image_url": "https://example.com/pb.jpg",
            "quantity_available": 12}
    script = [[{"type": "tool_calls", "tool_calls": [
        {"id": "tc-1", "name": "semantic_search", "arguments": {"query": "power bank", "top_k": 5}},
        {"id": "tc-2", "name": "search_products", "arguments": {"q": "power bank", "limit": 20}},
    ]},
    {"type": "finish", "reason": "tool_use"}],
    ]
    agent, vs, backend, model = _agent_for(products=[prod], vector_items=[], script=script)
    session = Session(session_id="smoke-contradiction", role=None)
    events = [ev async for ev in agent.run_turn(session, "power bank")]

    texts = [e.get("content") for e in events if e.get("type") == "text"]
    assert not any("couldn't find any results" in t for t in texts), \
        f"semantic_search miss leaked into transcript: {texts}"
    assert any("Found 1 product" in t for t in texts), \
        f"expected product-search hit summary in {texts}"
    directives = [e["directive"] for e in events if e.get("type") == "ui_directive"]
    grids = [d for d in directives if d["component"] == "ProductGrid"]
    assert any(len(g["props"]["items"]) >= 1 for g in grids), \
        f"no populated ProductGrid directive: {grids}"


async def test_semantic_miss_emitted_when_no_product_hit():
    """Counterpart: when there is no product hit, the semantic miss summary
    should still be emitted (not over-suppressed)."""
    script = [[{"type": "tool_calls", "tool_calls": [
        {"id": "tc-1", "name": "semantic_search", "arguments": {"query": "phone case", "top_k": 5}},
        {"id": "tc-2", "name": "search_products", "arguments": {"q": "phone case", "limit": 20}},
    ]},
    {"type": "finish", "reason": "tool_use"}],
    ]
    agent, vs, backend, model = _agent_for(products=[], vector_items=[], script=script)
    session = Session(session_id="smoke-miss", role=None)
    events = [ev async for ev in agent.run_turn(session, "phone case")]
    texts = [e.get("content") for e in events if e.get("type") == "text"]
    assert any("couldn't find any results" in t for t in texts), \
        f"semantic miss summary should appear when no hit: {texts}"




async def test_semantic_miss_suppressed_across_iterations():
    """Regression: when semantic_search misses in iteration 1 and
    search_products hits in iteration 2 (separate agent-loop iterations,
    not the same stream), the 'couldn't find any results' summary must
    still be suppressed because the per-turn buffer sees both summaries.
    """
    prod = {"id": "prod-50", "title": "Power Bank 20000mAh", "merchant_id": "m-1",
            "merchant_name": "Gadget Store", "category_id": "cat-3",
            "base_price_amount": 12000.0, "base_price_currency": "NGN",
            "status": "active", "image_url": "https://example.com/pb.jpg",
            "quantity_available": 12}
    script = [
        # Iteration 1: semantic_search only (miss - returns 0 items)
        [{"type": "tool_calls", "tool_calls": [
            {"id": "tc-1", "name": "semantic_search", "arguments": {"query": "power bank", "top_k": 5}},
        ]},
        {"type": "finish", "reason": "tool_use"}],
        # Iteration 2: search_products (hit - returns 1 product)
        [{"type": "tool_calls", "tool_calls": [
            {"id": "tc-2", "name": "search_products", "arguments": {"q": "power bank", "limit": 20}},
        ]},
        {"type": "finish", "reason": "tool_use"}],
        # Iteration 3: model finishes (no more tool calls)
        [{"type": "finish", "reason": "end_turn"}],
    ]
    agent, vs, backend, model = _agent_for(products=[prod], vector_items=[], script=script)
    session = Session(session_id="cross-iter-suppression", role=None)
    events = [ev async for ev in agent.run_turn(session, "power bank")]

    texts = [e.get("content") for e in events if e.get("type") == "text"]
    assert not any("couldn't find any results" in t for t in texts), \
        f"semantic_search miss leaked into transcript across iterations: {texts}"
    assert any("Found 1 product" in t for t in texts), \
        f"expected product-search hit summary in {texts}"
    directives = [e["directive"] for e in events if e.get("type") == "ui_directive"]
    grids = [d for d in directives if d["component"] == "ProductGrid"]
    assert any(len(g["props"]["items"]) >= 1 for g in grids), \
        f"no populated ProductGrid directive: {grids}"


async def test_signed_in_user_add_to_cart_not_prompted():
    """Regression for signed-in users being asked to sign in when adding to
    cart: a session carrying a user_token must bypass the action-gate so
    add_to_cart is executed, not turned into a sign-in request.
    """
    script = [[{"type": "tool_calls", "tool_calls": [
        {"id": "tc-1", "name": "add_to_cart", "arguments": {
            "variant_id": "v-1", "product_id": "prod-50", "product_name": "wallet",
            "quantity": 1, "unit_price_amount": 100, "unit_price_currency": "NGN",
        }},
    ]},
    {"type": "finish", "reason": "tool_use"}],
    ]
    agent, vs, backend, model = _agent_for(script=script)
    session = Session(session_id="smoke-addcart", role=None, user_token="Bearer test-token")
    events = [ev async for ev in agent.run_turn(session, "add this to my cart")]
    blob = json.dumps(events).lower()
    assert "sign_in" not in blob and "action_required" not in blob, \
        f"signed-in user was prompted to authenticate: {events}"
    contents = [e.get("content", "") for e in events]
    assert any("added to your cart" in t for t in contents), \
        f"add_to_cart summary missing: {contents}"
