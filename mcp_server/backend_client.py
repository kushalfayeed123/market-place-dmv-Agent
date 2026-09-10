"""Backend HTTP client — makes authenticated calls to the marketplace backend.

Every MCP tool call is just an authenticated HTTP call to the backend, so it
passes through the same rate limiter, RBAC, and idempotency layer as a
human-driven client. The user's access token is injected per-call and is NEVER
part of the text the model sees.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import httpx


class ToolError(Exception):
    """Raised when a backend call fails. Maps HTTP status to typed error kinds."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_kind: str = "unknown",
        response_body: dict | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_kind = error_kind
        self.response_body = response_body or {}


class BackendClient:
    """Async HTTP client for the backend API.

    Each call carries:
    - The user's Bearer token (injected per-call, never stored in the client)
    - A correlation_id (propagated to backend via X-Correlation-ID)
    - An Idempotency-Key for mutating tools (derived server-side)
    """

    def __init__(self, base_url: str, api_prefix: str = "/api/v1", timeout: float = 30.0):
        self._base = base_url.rstrip("/")
        self._prefix = api_prefix.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=f"{self._base}{self._prefix}",
            timeout=self._timeout,
            http2=True,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def call(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        path_params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated call to the backend.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE).
            path: Path relative to API prefix (may contain {param} placeholders).
            token: User's Bearer access token (None for public endpoints).
            idempotency_key: Idempotency-Key header for mutating tools.
            correlation_id: Correlation ID for tracing (generated if not provided).
            params: Query parameters.
            json_body: JSON request body.
            path_params: Values for {param} placeholders in the path.

        Returns:
            Parsed JSON response body.

        Raises:
            ToolError: On non-2xx response, with typed error_kind.
        """
        # Build the actual path
        actual_path = path
        if path_params:
            for key, value in path_params.items():
                actual_path = actual_path.replace(f"{{{key}}}", str(value))

        # Build headers
        headers: dict[str, str] = {
            "Accept": "application/json",
            "X-Correlation-ID": correlation_id or uuid.uuid4().hex,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        # Make the request
        response = await self._client.request(
            method=method.upper(),
            url=actual_path,
            params=params,
            json=json_body,
            headers=headers,
        )

        # Parse response
        try:
            body = response.json() if response.content else {}
        except Exception:
            body = {"raw": response.text}

        # Handle errors
        if response.status_code >= 400:
            error_kind = _classify_error(response.status_code, body)
            raise ToolError(
                message=body.get("detail", f"Backend returned {response.status_code}"),
                status_code=response.status_code,
                error_kind=error_kind,
                response_body=body,
            )

        return body


def _classify_error(status_code: int, body: dict) -> str:
    """Map HTTP status + body to a typed error kind."""
    if status_code == 401:
        return "unauthenticated"
    if status_code == 403:
        return "permission_denied"
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        return "conflict"  # e.g. idempotency in-progress
    if status_code == 422:
        return "validation_error"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "backend_error"
    return "unknown"
