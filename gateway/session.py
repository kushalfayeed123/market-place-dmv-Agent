"""Session management: per-user session state stored in Redis.

Holds the user's backend token (never shown to the model), identity, the turn counter,
the session cart (the backend has no cart resource yet) and a short conversation history
so follow-ups such as 'add the second one' work.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import redis.asyncio as redis

HISTORY_LIMIT = 12  # messages kept (user + assistant text only)


class Session:
    def __init__(
        self,
        session_id: str,
        user_token: str | None = None,
        user_id: str | None = None,
        role: str | None = None,
    ):
        self.session_id = session_id
        self.user_token = user_token
        self.user_id = user_id
        self.role = role
        self.turn_counter: int = 0
        self._cart: dict = {"items": []}
        self.history: list[dict[str, str]] = []

    @property
    def token(self) -> str | None:
        """Never surfaces in model messages or directive props."""
        return self.user_token

    def next_turn(self) -> int:
        self.turn_counter += 1
        return self.turn_counter

    @property
    def current_turn_id(self) -> int:
        return self.turn_counter

    # cart
    def get_cart(self) -> dict:
        return self._cart

    def set_cart(self, cart: dict) -> None:
        self._cart = cart

    def clear_cart(self) -> None:
        self._cart = {"items": []}

    # history
    def append_exchange(self, user_text: str, assistant_text: str) -> None:
        if user_text:
            self.history.append({"role": "user", "content": user_text[:1000]})
        if assistant_text:
            self.history.append({"role": "assistant", "content": assistant_text[:1000]})
        self.history = self.history[-HISTORY_LIMIT:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_token": self.user_token,
            "user_id": self.user_id,
            "role": self.role,
            "turn_counter": self.turn_counter,
            "cart": self._cart,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        s = cls(
            session_id=data["session_id"],
            user_token=data.get("user_token"),
            user_id=data.get("user_id"),
            role=data.get("role"),
        )
        s.turn_counter = data.get("turn_counter", 0)
        s._cart = data.get("cart", {"items": []})
        s.history = list(data.get("history", []))[-HISTORY_LIMIT:]
        return s


class SessionStore:
    """Redis-backed session store. The TTL is refreshed on every read and write."""

    def __init__(self, redis_client: redis.Redis, namespace: str = "agent", ttl: int = 7 * 24 * 3600):
        self._redis = redis_client
        self._ns = namespace
        self._ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"{self._ns}:session:{session_id}"

    async def create_session(
        self, user_token: str | None = None, user_id: str | None = None, role: str | None = None
    ) -> Session:
        session = Session(session_id=uuid.uuid4().hex, user_token=user_token, user_id=user_id, role=role)
        await self._save(session)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        key = self._key(session_id)
        data = await self._redis.get(key)
        if data is None:
            return None
        try:
            session = Session.from_dict(json.loads(data))
        except (json.JSONDecodeError, KeyError):
            return None
        await self._redis.expire(key, self._ttl)
        return session

    async def save_session(self, session: Session) -> None:
        await self._save(session)

    async def _save(self, session: Session) -> None:
        await self._redis.setex(self._key(session.session_id), self._ttl, json.dumps(session.to_dict()))

    async def delete_session(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
