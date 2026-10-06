"""Confirmation gate — server-side enforcement for mutating_financial tools.

A model can be prompted to "always ask before charging," but prompts are not
enforcement. The actual gate lives here: any tool tagged mutating_financial
called without a valid user_confirmed_token is refused. The token is minted
by the frontend, tied to an actual DOM click event on ConfirmationDialog.
"""

from __future__ import annotations

import time
from typing import Optional


class ConfirmationRequiredError(Exception):
    """Raised when a mutating_financial tool is called without a valid confirmation token."""
    pass


class ConfirmationGate:
    """Validates user_confirmed_tokens for mutating_financial tool calls.

    Tokens are short-lived, single-use, and bound to (session_id, turn_id, tool_name).
    They are minted by the frontend when the user clicks the actual Confirm button
    and stored in Redis by the Gateway before the tool call reaches the MCP server.
    """

    def __init__(self, redis_client, namespace: str = "agent", ttl: int = 300):
        self._redis = redis_client
        self._ns = namespace
        self._ttl = ttl

    def _key(self, session_id: str, turn_id: int, tool_name: str) -> str:
        return f"{self._ns}:confirmed_token:{session_id}:{turn_id}:{tool_name}"

    async def store_token(
        self,
        session_id: str,
        turn_id: int,
        tool_name: str,
        token: str,
    ) -> None:
        """Store a confirmed token (called by Gateway after frontend click)."""
        key = self._key(session_id, turn_id, tool_name)
        await self._redis.set(key, token, ex=self._ttl)

    async def consume_token(
        self,
        session_id: str,
        turn_id: int,
        tool_name: str,
        provided_token: str,
    ) -> bool:
        """Consume (verify + delete) a token. Returns True if valid.

        Single-use: the key is deleted immediately upon successful verification,
        so a replayed token cannot be reused.

        In production with a real Redis server, a Lua script provides an atomic
        get-and-delete that is immune to race conditions. Some test doubles
        (e.g. fakeredis) do not implement EVAL; in that case we fall back to
        a sequential get+delete which is correct for single-threaded test
        execution while preserving the exact same verification semantics.
        """
        key = self._key(session_id, turn_id, tool_name)
        # Prefer the atomic Lua script (real Redis); fall back if unsupported
        lua_script = """
        local current = redis.call('get', KEYS[1])
        if current == false then
            return 0
        end
        if current == ARGV[1] then
            redis.call('del', KEYS[1])
            return 1
        end
        return 0
        """
        try:
            result = await self._redis.eval(lua_script, 1, key, provided_token)
            return bool(result)
        except Exception:
            # Fallback for Redis implementations that do not support EVAL
            # (e.g. fakeredis in test mode). Single-threaded test execution
            # means we do not need the Lua atomicity guarantee.
            stored_token = await self._redis.get(key)
            if stored_token is None:
                return False
            if stored_token == provided_token:
                await self._redis.delete(key)
                return True
            return False

    async def check_confirmation(
        self,
        session_id: str,
        turn_id: int,
        tool_name: str,
        provided_token: Optional[str],
    ) -> None:
        """Raise ConfirmationRequiredError if no valid token is provided.

        This is the single most important security check in the agent layer.
        Called by the MCP server before executing any mutating_financial tool.
        """
        if not provided_token:
            raise ConfirmationRequiredError(
                f"Tool '{tool_name}' requires user confirmation. "
                "A user_confirmed_token must be provided from a real frontend button click."
            )
        valid = await self.consume_token(session_id, turn_id, tool_name, provided_token)
        if not valid:
            raise ConfirmationRequiredError(
                f"Invalid or expired confirmation token for tool '{tool_name}'. "
                "The user must click Confirm again."
            )
