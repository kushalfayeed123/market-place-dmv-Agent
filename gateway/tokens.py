"""Access-token helpers."""

from __future__ import annotations


def normalize_token(raw: str | None) -> str | None:
    """Return the bare access token.

    The frontend forwards 'Bearer <token>' as `user_token`, while BackendClient
    adds its own 'Bearer ' prefix, which produced 'Bearer Bearer <token>' and a
    401 for every signed-in user. Accept both forms and always return the bare token.
    """
    if not raw:
        return None
    token = raw.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None
