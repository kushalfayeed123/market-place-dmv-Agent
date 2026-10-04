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
SYSTEM_PROMPT = open(os.path.join(HERE, "..", "prompts", "system.md"), encoding="utf-8").read()


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

    # Real schemas advertised (so the model can pick/call tools accurately).
    assert "query" in parameters_for("semantic_search")["properties"]
    assert parameters_for("semantic_search")["required"] == ["query"]
    assert "q" in parameters_for("search_products")["properties"]

    # semantic_search actually fired against the vector store.
    assert vs.queries and "red leather wallet" in vs.queries[0]
    # Combination of tools: the KB doc referenced a product id, so the model
    # followed up with get_product_detail(product_id=...).
    assert backend.last_pp.get("product_id") == "prod-50"
    assert backend.calls == 1
    # The agentic loop re-prompted after each tool result.
    assert model.calls == 3
    assert "tool" in [m.get("role") for m in model.received[1]]  # KB result fed back

    directives = [e["directive"] for e in events if e.get("type") == "ui_directive"]
    assert any(d["component"] == "ProductGrid" for d in directives)
    detail = next(d for d in directives if d["component"] == "ProductDetail")
    assert detail["props"]["name"] == "Red Leather Wallet"
    grid = next(d for d in directives if d["component"] == "ProductGrid")
    assert grid["props"]["items"][0]["text"] == vs._items[0]["text"]  # untrusted data surfaced

    blob = json.dumps(events).lower()
    assert "access_token" not in blob and "user_token" not in blob


async def test_semantic_search_without_vector_store_degrades_gracefully():
    agent, _, _, _ = _agent()
    agent._vector_store = None
    agent._redis = None
    session = Session(session_id="smoke-grace", role=None)
    events = [ev async for ev in agent.run_turn(session, "anything")]
    texts = [e.get("content") for e in events if e.get("type") == "text"]
    assert any("anything relevant" in t for t in texts)  # empty KB -> friendly msg, no crash
