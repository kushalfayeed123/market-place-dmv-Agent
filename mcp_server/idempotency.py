"""Idempotency key derivation — server-side, outside model control.

LLM tool-calling loops DO retry, and a naive retry on a payment tool must never
double-charge. For every tool tagged mutating_financial, the MCP server derives
a stable Idempotency-Key from the user's actual intent:

    key = sha256(user_id + conversation_turn_id + normalized_request_body)

If the agent calls the same tool again for what is semantically the same user
turn (a retry after a timeout, a model re-attempt after a transient error), the
derived key is identical, so the backend's idempotency layer returns the cached
result instead of executing twice.

If the user issues a genuinely NEW request (e.g., checks out again after adding
a new item), the turn id changes, producing a new key — correctly treated as a
new transaction.

The model is never asked to generate or manage this key itself; key generation
is entirely outside model control.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def derive_idempotency_key(
    user_id: str | None,
    turn_id: int,
    tool_name: str,
    body: dict[str, Any] | None,
) -> str:
    """Derive a stable idempotency key for a mutating tool call.

    Args:
        user_id: The authenticated user's ID (None for anonymous — but anonymous
                 users should never reach a mutating_financial tool).
        turn_id: The Gateway's incrementing turn counter for this session.
        tool_name: The tool being called.
        body: The request body (will be canonicalized).

    Returns:
        A hex-encoded SHA-256 hash to send as the Idempotency-Key header.
    """
    # Canonicalize the body: sort keys, strip whitespace, use compact JSON
    canonical_body = _canonicalize(body) if body else ""

    # Build the dedup string
    dedup_string = f"{user_id or 'anonymous'}:{turn_id}:{tool_name}:{canonical_body}"

    return hashlib.sha256(dedup_string.encode("utf-8")).hexdigest()


def _canonicalize(obj: Any) -> str:
    """Produce a canonical JSON representation for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
