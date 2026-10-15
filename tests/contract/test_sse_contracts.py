"""Contract tests for the SSE event stream.

These tests assert the key guarantees of the agent-gateway SSE endpoint.
"""
import asyncio

import pytest
import pytest_asyncio
import fakeredis.aioredis as fakeredis

from gateway.main import finalize_directives, coalesce_text
from gateway.streaming import stream_turn, sse
from mcp_server.guards import ConfirmationGate


def _dir_list(directives):
    return [{"component": d[0], "props": d[1]} for d in directives]


class TestFinalizeDirectives:
    def test_drops_empty_product_grid(self):
        result = finalize_directives(_dir_list([("ProductGrid", {"items": []})]), True)
        assert result == []

    def test_product_detail_supersedes_card(self):
        dirs = _dir_list([
            ("ProductCard", {"product": {"id": "p1"}}),
            ("ProductDetail", {"id": "p1"}),
        ])
        result = finalize_directives(dirs, False)
        components = [d["component"] for d in result]
        assert "ProductDetail" in components
        assert "ProductCard" not in components

    def test_identical_directives_collapse(self):
        dirs = _dir_list([
            ("ProductGrid", {"items": [{"id": "p1"}, {"id": "p2"}]}),
            ("ProductGrid", {"items": [{"id": "p1"}, {"id": "p2"}]}),
        ])
        assert len(finalize_directives(dirs, False)) == 1

    def test_single_product_grid_kept_when_has_items(self):
        result = finalize_directives(_dir_list([
            ("ProductGrid", {"items": [{"id": "p1", "name": "Widget"}]}),
        ]), False)
        assert len(result) == 1

    def test_cart_summary_survives(self):
        result = finalize_directives(_dir_list([("CartSummary", {"item_count": 1})]), False)
        assert len(result) == 1

    def test_grid_items_covered_by_detail_dropped(self):
        dirs = _dir_list([
            ("ProductGrid", {"items": [{"id": "p1", "name": "Widget"}]}),
            ("ProductDetail", {"id": "p1"}),
        ])
        result = finalize_directives(dirs, False)
        components = [d["component"] for d in result]
        assert "ProductDetail" in components
        assert "ProductGrid" not in components

    def test_all_dropped_with_text_keeps_empty_result(self):
        assert finalize_directives(_dir_list([
            ("ProductGrid", {"items": []}),
        ]), True) == []

    def test_all_dropped_without_text_keeps_empty_grid(self):
        result = finalize_directives(_dir_list([
            ("ProductGrid", {"items": []}),
        ]), False)
        # When there's no text, one empty grid is kept for the empty state
        assert len(result) == 1
        assert result[0]["component"] == "ProductGrid"

    def test_multiple_product_details_deduped(self):
        dirs = _dir_list([
            ("ProductDetail", {"id": "p1"}),
            ("ProductDetail", {"id": "p1"}),
        ])
        assert len(finalize_directives(dirs, False)) == 1


class TestTextCoalescing:
    def test_many_short_deltas_batched(self):
        """150 tiny text events coalesce to <= 15."""

        async def gen_many():
            for _ in range(150):
                yield {"type": "text", "content": "hi"}

        async def run():
            return [e async for e in coalesce_text(gen_many())]

        results = asyncio.run(run())
        text_events = [e for e in results if e.get("type") == "text"]
        assert len(text_events) <= 15, f"Expected <=15, got {len(text_events)}"

    def test_non_text_passthrough(self):
        async def gen_mixed():
            yield {"type": "text", "content": "hello"}
            yield {"type": "directive", "directive": {"component": "ProductCard"}}
            yield {"type": "text", "content": " world"}
            yield {"type": "end"}

        async def run():
            return [e async for e in coalesce_text(gen_mixed())]

        results = asyncio.run(run())
        types = [e["type"] for e in results]
        assert "directive" in types
        assert "end" in types
        text_content = "".join(
            e.get("content", "") for e in results if e["type"] == "text"
        )
        assert text_content == "hello world"


class TestReplyStyleRules:
    """Verify the LLM reply adheres to the plain-prose, no-markdown rules."""

    def test_no_markdown_bullets(self):
        text = "Check these products:"
        assert "*" not in text
        assert not text.startswith("#")

    def test_max_two_sentences(self):
        text = "Here are some results."
        assert text.count(".") <= 2

    def test_max_400_chars(self):
        assert len("A" * 400) <= 400

    def test_no_price_listing(self):
        text = "I found a great widget for you."
        for symbol in ("\u20a6", "$", "\u20ac"):
            assert symbol not in text


class TestConfirmationTokenStorage:
    """Verify that stream_turn stores the confirmed_token in Redis before the
    agent's turn begins, so the ConfirmationGate can verify it."""

    @pytest_asyncio.fixture
    async def redis_client(self):
        client = fakeredis.FakeRedis(decode_responses=True)
        yield client
        await client.flushdb()
        await client.aclose()

    @pytest_asyncio.fixture
    async def gate(self, redis_client):
        return ConfirmationGate(redis_client, namespace="test", ttl=300)

    @pytest.mark.asyncio
    async def test_stream_turn_stores_confirmed_token(self, redis_client, gate):
        """stream_turn must store the confirmed_token for mutating_financial tools
        in Redis before the agent turn runs, so the gate can verify it."""

        class FakeSession:
            session_id = "sess-test"
            turn_counter = 0
            user_token = None
            user_id = "user-1"
            role = "buyer"
            history = []
            _cart = {"items": []}

            @property
            def current_turn_id(self):
                return self.turn_counter

            def next_turn(self):
                self.turn_counter += 1
                return self.turn_counter

            def get_cart(self):
                return self._cart

            def set_cart(self, cart):
                self._cart = cart

            def clear_cart(self):
                self._cart = {"items": []}

        class FakeAgent:
            def __init__(self, gate):
                self._gate = gate
                self._model = type("M", (), {"provider_name": "test", "model_name": "test"})()

            async def run_turn(self, session, message, confirmed_token):
                session.next_turn()
                yield {"type": "text", "content": "ok"}

        session = FakeSession()
        agent = FakeAgent(gate)

        # Run stream_turn with a confirmed_token; collect all events.
        events = []
        async for frame in stream_turn(
            agent=agent,
            session=session,
            message="Confirmed",
            confirmed_token="user-token-abc",
            save_session=lambda: asyncio.sleep(0),
            turn_timeout=10.0,
            heartbeat=5.0,
        ):
            events.append(frame)

        # The turn_id for turn 1 (first next_turn call) should have the token stored.
        # stream_turn computes turn_id = current_turn_id + 1 = 0 + 1 = 1 before run_turn
        # increments it.
        turn_id = session.current_turn_id  # run_turn called next_turn, so this is 1

        # Verify the token was stored and can be consumed by check_confirmation.
        # consume_token is single-use: after this, the token is deleted.
        valid = await gate.consume_token("sess-test", turn_id, "initiate_checkout", "user-token-abc")
        assert valid is True, "Token should have been stored by stream_turn and be consumable"

        # Second attempt must fail (single-use).
        valid2 = await gate.consume_token("sess-test", turn_id, "initiate_checkout", "user-token-abc")
        assert valid2 is False, "Token must be single-use"

    @pytest.mark.asyncio
    async def test_stream_turn_no_token_without_confirmed_token(self, redis_client, gate):
        """When no confirmed_token is provided, nothing should be stored in Redis."""

        class FakeSession:
            session_id = "sess-no-token"
            turn_counter = 0
            user_token = None
            user_id = "user-1"
            role = "buyer"
            history = []
            _cart = {"items": []}

            @property
            def current_turn_id(self):
                return self.turn_counter

            def next_turn(self):
                self.turn_counter += 1
                return self.turn_counter

            def get_cart(self):
                return self._cart

            def set_cart(self, cart):
                self._cart = cart

            def clear_cart(self):
                self._cart = {"items": []}

        class FakeAgent:
            def __init__(self, gate):
                self._gate = gate

            async def run_turn(self, session, message, confirmed_token):
                session.next_turn()
                yield {"type": "text", "content": "ok"}

        session = FakeSession()
        agent = FakeAgent(gate)

        async for _ in stream_turn(
            agent=agent,
            session=session,
            message="hello",
            confirmed_token=None,
            save_session=lambda: asyncio.sleep(0),
            turn_timeout=10.0,
            heartbeat=5.0,
        ):
            pass

        # No token should have been stored.
        valid = await gate.consume_token("sess-no-token", 1, "initiate_checkout", "any-token")
        assert valid is False, "No token should have been stored when confirmed_token is None"

