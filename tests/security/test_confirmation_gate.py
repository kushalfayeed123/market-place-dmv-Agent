"""Confirmation gate tests — the single most important tests in the agent layer.

A mutating_financial tool MUST be rejected without a valid user_confirmed_token.
This is server-side enforcement, not a prompt instruction.
"""

import pytest
import pytest_asyncio
import redis.asyncio as redis
import fakeredis.aioredis as fakeredis

from mcp_server.guards import ConfirmationGate, ConfirmationRequiredError


@pytest_asyncio.fixture
async def redis_client():
    """Create a test Redis client.

    Tries a real Redis at localhost:6379 first. If it is unavailable
    (common in local / CI environments without Redis installed), falls
    back to an in-memory fakeredis instance so the confirmation-gate
    tests can still exercise the full ``store_token`` / ``consume_token``
    flow.
    """
    try:
        client = redis.from_url("redis://localhost:6379/15", decode_responses=True)
        await client.ping()
    except Exception:
        client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    await client.flushdb()
    await client.aclose()


@pytest_asyncio.fixture
async def gate(redis_client):
    """Create a ConfirmationGate backed by test Redis."""
    return ConfirmationGate(redis_client, namespace="test", ttl=300)


@pytest.mark.asyncio
class TestConfirmationGate:

    async def test_mutating_financial_tool_rejected_without_token(self, gate):
        """Calling a financial tool with no token MUST raise ConfirmationRequiredError."""
        with pytest.raises(ConfirmationRequiredError):
            await gate.check_confirmation("session-1", 1, "initiate_checkout", None)

    async def test_mutating_financial_tool_rejected_with_wrong_token(self, gate):
        """Calling a financial tool with an invalid token MUST raise ConfirmationRequiredError."""
        with pytest.raises(ConfirmationRequiredError):
            await gate.check_confirmation("session-1", 1, "initiate_checkout", "wrong-token")

    async def test_mutating_financial_tool_rejected_with_expired_token(self, gate):
        """A token that was never stored is treated as expired/invalid."""
        with pytest.raises(ConfirmationRequiredError):
            await gate.check_confirmation("session-1", 1, "initiate_checkout", "expired-token")

    async def test_valid_token_accepted(self, gate):
        """A valid stored token MUST be accepted."""
        await gate.store_token("session-1", 1, "initiate_checkout", "valid-token-123")
        # Should not raise
        await gate.check_confirmation("session-1", 1, "initiate_checkout", "valid-token-123")

    async def test_token_is_single_use(self, gate):
        """A token MUST be consumed (deleted) after successful verification."""
        await gate.store_token("session-1", 1, "initiate_checkout", "single-use-token")
        # First use succeeds
        await gate.check_confirmation("session-1", 1, "initiate_checkout", "single-use-token")
        # Second use fails (token consumed)
        with pytest.raises(ConfirmationRequiredError):
            await gate.check_confirmation("session-1", 1, "initiate_checkout", "single-use-token")

    async def test_token_bound_to_session_turn_and_tool(self, gate):
        """A token for one session/turn/tool must not work for another."""
        await gate.store_token("session-1", 1, "initiate_checkout", "bound-token")
        # Wrong session
        with pytest.raises(ConfirmationRequiredError):
            await gate.check_confirmation("session-2", 1, "initiate_checkout", "bound-token")
        # Wrong turn
        with pytest.raises(ConfirmationRequiredError):
            await gate.check_confirmation("session-1", 2, "initiate_checkout", "bound-token")
        # Wrong tool
        with pytest.raises(ConfirmationRequiredError):
            await gate.check_confirmation("session-1", 1, "request_refund", "bound-token")

    async def test_all_financial_tools_require_confirmation(self, gate):
        """Every mutating_financial tool must be rejected without a token."""
        financial_tools = [
            "initiate_checkout",
            "process_payment",
            "request_refund",
            "review_kyc",
            "request_payout",
        ]
        for tool in financial_tools:
            with pytest.raises(ConfirmationRequiredError):
                await gate.check_confirmation("session-1", 1, tool, None)
