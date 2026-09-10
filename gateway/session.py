"""Session management — per-user session state stored in Redis.

Each session holds:
- user_access_token: the backend Bearer token (never exposed to the model)
- user_id, role: for tool visibility and address
- turn_counter: increments each turn, used for idempotency key derivation
- cart: session-scoped cart (the backend has no cart resource)
- confirmed_tokens: short-lived tokens from frontend button clicks
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

import redis.asyncio as redis


class Session:
    """A single user session."""

    def __init__(
        self,
        session_id: str,
        user_token: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
    ):
        self.session_id = session_id
        self.user_token = user_token
        self.user_id = user_id
        self.role = role
        self.turn_counter: int = 0
        self._cart: dict = {"items": []}

    def next_turn(self) -> int:
        """Increment and return the turn counter."""
        self.turn_counter += 1
        return self.turn_counter

    @property
    def current_turn_id(self) -> int:
        return self.turn_counter

    # ── Cart operations ─────────────────────────────────────────────────
    def get_cart(self) -> dict:
        return self._cart

    def set_cart(self, cart: dict) -> None:
        self._cart = cart

    def clear_cart(self) -> None:
        self._cart = {"items": []}

    # ── Serialization ──────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_token": self.user_token,
            "user_id": self.user_id,
            "role": self.role,
            "turn_counter": self.turn_counter,
            "cart": self._cart,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        s = cls(
            session_id=data["session_id"],
            user_token=data.get("user_token"),
            user_id=data.get("user_id"),
            role=data.get("role"),
        )
        s.turn_counter = data.get("turn_counter", 0)
        s._cart = data.get("cart", {"items": []})
        return s


class SessionStore:
    """Redis-backed session store."""

    def __init__(self, redis_client: redis.Redis, namespace: str = "agent", ttl: int = 3600):
        self._redis = redis_client
        self._ns = namespace
        self._ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"{self._ns}:session:{session_id}"

    async def create_session(
        self,
        user_token: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Session:
        """Create a new session."""
        session_id = uuid.uuid4().hex
        session = Session(
            session_id=session_id,
            user_token=user_token,
            user_id=user_id,
            role=role,
        )
        await self._save(session)
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Load a session by ID."""
        key = self._key(session_id)
        data = await self._redis.get(key)
        if data is None:
            return None
        try:
            return Session.from_dict(json.loads(data))
        except (json.JSONDecodeError, KeyError):
            return None

    async def save_session(self, session: Session) -> None:
        """Persist a session."""
        await self._save(session)

    async def _save(self, session: Session) -> None:
        key = self._key(session.session_id)
        await self._redis.setex(key, self._ttl, json.dumps(session.to_dict()))

    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        await self._redis.delete(self._key(session_id))
