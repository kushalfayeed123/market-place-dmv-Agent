"""DebugTracer — structured event log for viewing the agent's internal processing.

Exposes exactly what the agent did: model calls, tool calls, guard decisions,
directives emitted, and errors. Available via:
1. /debug/sessions/{id}/trace — full structured trace (postmortem)
2. /debug/sessions/{id}/events — SSE stream mirroring the trace live

Safety: never emits the user's access_token, raw card data, or full PII.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from typing import Any, Optional

import redis.asyncio as redis


class DebugEvent:
    """A single debug event."""

    def __init__(
        self,
        event_type: str,
        data: dict[str, Any],
        session_id: str,
        turn_id: int,
        timestamp: float | None = None,
    ):
        self.id = uuid.uuid4().hex[:12]
        self.event_type = event_type
        self.data = _sanitize(data)
        self.session_id = session_id
        self.turn_id = turn_id
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "data": self.data,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
        }


class DebugTracer:
    """Records and retrieves debug events for a session."""

    def __init__(self, redis_client: redis.Redis, namespace: str = "agent", enabled: bool = True):
        self._redis = redis_client
        self._ns = namespace
        self._enabled = enabled
        self._active_buffers: dict[str, list[DebugEvent]] = defaultdict(list)

    def _key(self, session_id: str) -> str:
        return f"{self._ns}:debug:{session_id}"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(
        self,
        event_type: str,
        data: dict[str, Any],
        session_id: str,
        turn_id: int = 0,
    ) -> DebugEvent | None:
        """Log a debug event. Returns the event, or None if debug is disabled."""
        if not self._enabled:
            return None

        event = DebugEvent(
            event_type=event_type,
            data=data,
            session_id=session_id,
            turn_id=turn_id,
        )
        self._active_buffers[session_id].append(event)
        return event

    # ── Convenience loggers ──────────────────────────────────────────────
    def log_turn_start(self, session_id: str, turn_id: int, user_message: str) -> None:
        self.log("turn.start", {"user_message": user_message[:500]}, session_id, turn_id)

    def log_intent(self, session_id: str, turn_id: int, intent: str, confidence: float | None = None) -> None:
        self.log("intent.classified", {"intent": intent, "confidence": confidence}, session_id, turn_id)

    def log_tool_selected(self, session_id: str, turn_id: int, tool_name: str, args: dict) -> None:
        self.log("tool.selected", {"tool_name": tool_name, "args": _sanitize(args)}, session_id, turn_id)

    def log_idempotency_key(self, session_id: str, turn_id: int, tool_name: str, key: str) -> None:
        self.log("idempotency.key_derived", {"tool_name": tool_name, "key": key[:16] + "..."}, session_id, turn_id)

    def log_backend_call(
        self, session_id: str, turn_id: int, method: str, path: str,
        status_code: int | None, latency_ms: float,
    ) -> None:
        self.log("backend.call", {
            "method": method, "path": path,
            "status_code": status_code, "latency_ms": round(latency_ms, 1),
        }, session_id, turn_id)

    def log_tool_result(self, session_id: str, turn_id: int, tool_name: str, result_summary: dict) -> None:
        self.log("tool.result", {"tool_name": tool_name, "result_summary": result_summary}, session_id, turn_id)

    def log_confirmation(
        self, session_id: str, turn_id: int, tool_name: str,
        required: bool, granted: bool, token_id: str | None = None,
    ) -> None:
        self.log("guard.confirmation", {
            "tool_name": tool_name, "required": required,
            "granted": granted, "token_id": token_id[:8] + "..." if token_id else None,
        }, session_id, turn_id)

    def log_directive(self, session_id: str, turn_id: int, component: str, props_summary: dict) -> None:
        self.log("directive.emitted", {"component": component, "props_summary": props_summary}, session_id, turn_id)

    def log_model_call(self, session_id: str, turn_id: int, provider: str, model: str) -> None:
        self.log("model.call", {"provider": provider, "model": model}, session_id, turn_id)

    def log_model_usage(self, session_id: str, turn_id: int, usage: dict) -> None:
        self.log("model.usage", {"usage": usage}, session_id, turn_id)

    def log_error(self, session_id: str, turn_id: int, error_type: str, message: str) -> None:
        self.log("error", {"error_type": error_type, "message": message[:500]}, session_id, turn_id)

    def log_turn_end(self, session_id: str, turn_id: int, reason: str) -> None:
        self.log("turn.end", {"reason": reason}, session_id, turn_id)


    # ── Retrieval ───────────────────────────────────────────────────────
    def get_trace(self, session_id: str) -> list[dict[str, Any]]:
        """Get the full debug trace for a session."""
        events = self._active_buffers.get(session_id, [])
        return [e.to_dict() for e in events]

    def get_events_for_turn(self, session_id: str, turn_id: int) -> list[dict[str, Any]]:
        """Get debug events for a specific turn."""
        events = self._active_buffers.get(session_id, [])
        return [e.to_dict() for e in events if e.turn_id == turn_id]

    async def persist_trace(self, session_id: str, ttl: int = 3600) -> None:
        """Persist the trace to Redis (for postmortem analysis)."""
        trace = self.get_trace(session_id)
        if trace:
            key = self._key(session_id)
            await self._redis.setex(key, ttl, json.dumps(trace))

    async def load_trace(self, session_id: str) -> list[dict[str, Any]]:
        """Load a persisted trace from Redis."""
        key = self._key(session_id)
        data = await self._redis.get(key)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return []
        return []


def _sanitize(data: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive fields from debug data."""
    sensitive_keys = {"access_token", "token", "password", "secret", "card_number", "cvv"}
    sanitized = {}
    for k, v in data.items():
        if k.lower() in sensitive_keys:
            sanitized[k] = "***REDACTED***"
        elif isinstance(v, dict):
            sanitized[k] = _sanitize(v)
        else:
            sanitized[k] = v
    return sanitized
