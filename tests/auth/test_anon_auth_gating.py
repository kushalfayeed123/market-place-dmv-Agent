"""Tests for anonymous-user auth gating.

Keyword heuristics supplement the LLM intent classifier so that:
  1. Anonymous users can ALWAYS search / browse / manage their cart.
  2. Anonymous users are ALWAYS prompted to sign in for authenticated actions.
"""
import os

import pytest

from gateway.agent import Agent, _is_anon_safe_query, _requires_auth_action
from gateway.debug import DebugTracer
from gateway.session import Session
from mcp_server.guards import ConfirmationGate
from schemas.components import ComponentName

HERE = os.path.dirname(__file__)
_PARENT = os.path.dirname(os.path.dirname(HERE))
SYSTEM_PROMPT = open(os.path.join(_PARENT, "prompts", "system.md"), encoding="utf-8").read()


class MisclassifyingModel:
    """Fake model whose classify returns a caller-supplied intent."""

    def __init__(self, classify_result):
        self._classify_result = classify_result
        self.calls = 0

    @property
    def provider_name(self):
        return "stub"

    @property
    def model_name(self):
        return "misclassifying-stub"

    async def stream(self, messages, tools, system_prompt, temperature=0.7, max_tokens=4096):
        yield {"type": "finish", "reason": "end_turn"}

    async def classify(self, message, system_prompt, valid_intents, temperature=0.0):
        return self._classify_result


class _SilentBackend:
    """A backend that records calls but never returns products."""

    def __init__(self):
        self.calls = 0

    async def call(self, method, path, *, token=None, params=None, path_params=None,
                   json_body=None, idempotency_key=None, correlation_id=None):
        self.calls += 1
        return {"items": []}


def _make_agent(classify_intent: str, classify_conf: float = 0.99):
    """Build an Agent with a FakeModel that misclassifies to the given intent."""
    model = MisclassifyingModel({
        "intent": classify_intent,
        "confidence": classify_conf,
        "reasoning": "stub",
        "error": None,
    })
    backend = _SilentBackend()
    return Agent(
        backend_client=backend,
        gate=ConfirmationGate(redis_client=None, namespace="agent"),
        model_backend=model,
        debug_tracer=DebugTracer(redis_client=None, namespace="agent", enabled=False),
        system_prompt=SYSTEM_PROMPT,
        vector_store=None,
        vector_namespace="agent",
    ), backend


def _get_components(events):
    """Extract component names from ui_directive events in *events*."""
    directives = [e.get("directive") for e in events if e.get("type") == "ui_directive"]
    return [d["component"] for d in directives if d]


# -- Keyword heuristic unit tests --

class TestAnonSafeQueryDetection:
    """Verify _is_anon_safe_query identifies search/browse/cart queries."""

    def test_search_query_detected(self):
        assert _is_anon_safe_query("search for shoes") is True
        assert _is_anon_safe_query("Find me a laptop") is True
        assert _is_anon_safe_query("show me products in electronics") is True

    def test_browsing_query_detected(self):
        assert _is_anon_safe_query("browse all categories") is True
        assert _is_anon_safe_query("what do you have?") is True
        assert _is_anon_safe_query("how many items are there") is True

    def test_cart_operation_detected(self):
        assert _is_anon_safe_query("add to cart") is True
        assert _is_anon_safe_query("add this to my cart") is True
        assert _is_anon_safe_query("view my cart") is True
        assert _is_anon_safe_query("remove from cart") is True

    def test_non_safe_query_returns_false(self):
        assert _is_anon_safe_query("buy this now") is False
        assert _is_anon_safe_query("checkout please") is False
        assert _is_anon_safe_query("process payment") is False


class TestRequiresAuthAction:
    """Verify _requires_auth_action identifies auth-requiring actions."""

    def test_buy_detected(self):
        assert _requires_auth_action("buy this phone") is True
        assert _requires_auth_action("I want to purchase this") is True

    def test_checkout_detected(self):
        assert _requires_auth_action("checkout please") is True
        assert _requires_auth_action("I want to checkout") is True

    def test_refund_detected(self):
        assert _requires_auth_action("refund this order") is True
        assert _requires_auth_action("I need a refund") is True

    def test_payment_detected(self):
        assert _requires_auth_action("pay for this") is True
        assert _requires_auth_action("process payment") is True

    def test_non_auth_query_returns_false(self):
        assert _requires_auth_action("search for shoes") is False
        assert _requires_auth_action("what is the price") is False
        assert _requires_auth_action("add to cart") is False


# -- Agent-level auth gating tests --

class TestAnonymousAuthGating:
    """End-to-end tests of the agent turn loop for anonymous users.

    Uses MisclassifyingModel to simulate an LLM that gets intent
    classification WRONG -- keyword heuristics must correct it.
    """

    @pytest.mark.asyncio
    async def test_anonymous_search_not_blocked_when_model_says_action(self):
        """Search misclassified as 'action' must still be allowed for anonymous."""
        agent, _b = _make_agent("action")
        session = Session(session_id="anon-search", role=None)
        events = [ev async for ev in agent.run_turn(session, "search for red shoes")]
        assert ComponentName.SignInPrompt not in _get_components(events)

    @pytest.mark.asyncio
    async def test_anonymous_search_not_blocked_when_model_says_info(self):
        """Normal case: model correctly classifies search as 'info'."""
        agent, _b = _make_agent("info")
        session = Session(session_id="anon-search-ok", role=None)
        events = [ev async for ev in agent.run_turn(session, "search for red shoes")]
        assert ComponentName.SignInPrompt not in _get_components(events)

    @pytest.mark.asyncio
    async def test_anonymous_buy_prompted_when_model_says_info(self):
        """Buy misclassified as 'info' must still trigger SignInPrompt."""
        agent, _b = _make_agent("info")
        session = Session(session_id="anon-buy", role=None)
        events = [ev async for ev in agent.run_turn(session, "I want to buy this phone")]
        assert ComponentName.SignInPrompt in _get_components(events)

    @pytest.mark.asyncio
    async def test_anonymous_buy_prompted_when_model_says_action(self):
        """Normal case: model correctly classifies buy as 'action'."""
        agent, _b = _make_agent("action")
        session = Session(session_id="anon-buy-ok", role=None)
        events = [ev async for ev in agent.run_turn(session, "buy this phone")]
        assert ComponentName.SignInPrompt in _get_components(events)

    @pytest.mark.asyncio
    async def test_anonymous_add_to_cart_not_blocked_when_model_says_action(self):
        """add_to_cart is anon-safe -- must not be blocked even if model says 'action'."""
        agent, _b = _make_agent("action")
        session = Session(session_id="anon-cart", role=None)
        events = [ev async for ev in agent.run_turn(session, "add this to my cart")]
        assert ComponentName.SignInPrompt not in _get_components(events)

    @pytest.mark.asyncio
    async def test_anonymous_checkout_prompted_when_model_says_info(self):
        """checkout requires auth -- must trigger SignInPrompt even if model says 'info'."""
        agent, _b = _make_agent("info")
        session = Session(session_id="anon-checkout", role=None)
        events = [ev async for ev in agent.run_turn(session, "checkout please")]
        assert ComponentName.SignInPrompt in _get_components(events)

    @pytest.mark.asyncio
    async def test_anonymous_refund_prompted_when_model_says_info(self):
        """refund requires auth -- must trigger SignInPrompt even if model says 'info'."""
        agent, _b = _make_agent("info")
        session = Session(session_id="anon-refund", role=None)
        events = [ev async for ev in agent.run_turn(session, "refund this order")]
        assert ComponentName.SignInPrompt in _get_components(events)

    @pytest.mark.asyncio
    async def test_anonymous_order_status_prompted_when_model_says_info(self):
        """Order status is user-scoped -- must trigger SignInPrompt for anonymous."""
        agent, _b = _make_agent("info")
        session = Session(session_id="anon-order", role=None)
        events = [ev async for ev in agent.run_turn(session, "what is my order status")]
        assert ComponentName.SignInPrompt in _get_components(events)

    @pytest.mark.asyncio
    async def test_signed_in_user_not_blocked_even_for_actions(self):
        """A signed-in user (has token) must bypass the auth gate entirely."""
        agent, _b = _make_agent("action")
        session = Session(session_id="authed-action", role="buyer",
                          user_token="Bearer test-token-123")
        events = [ev async for ev in agent.run_turn(session, "buy this phone")]
        assert ComponentName.SignInPrompt not in _get_components(events)

    @pytest.mark.asyncio
    async def test_how_to_checkout_is_informational(self):
        """'How do I checkout?' is informational -- must NOT trigger SignInPrompt."""
        agent, _b = _make_agent("action")
        session = Session(session_id="howto-checkout", role=None)
        events = [ev async for ev in agent.run_turn(session, "how do I checkout")]
        assert ComponentName.SignInPrompt not in _get_components(events)
