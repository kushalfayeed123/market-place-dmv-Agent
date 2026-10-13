"""Backend HTTP client: authenticated calls to the marketplace backend.

Hardened for deployment: follows redirects (http -> https, trailing slash), has connect
and read timeouts, retries idempotent GETs on connection errors and 502/503/504 (cold
start), never sends a doubled 'Bearer', and turns every failure into a typed ToolError
instead of an empty result.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

import httpx

logger = logging.getLogger("mcp.backend")


class ToolError(Exception):
    """A backend call failed. `error_kind` is machine-readable; `message` is safe to show."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_kind: str = "unknown",
        response_body: dict | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_kind = error_kind
        self.response_body = response_body or {}


def _normalize_token(raw: str | None) -> str | None:
    if not raw:
        return None
    token = raw.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None


def _http2_available() -> bool:
    try:
        import h2  # noqa: F401

        return True
    except ImportError:
        return False


class BackendClient:
    def __init__(
        self,
        base_url: str,
        api_prefix: str = "/api/v1",
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._base = base_url.rstrip("/")
        self._prefix = api_prefix.rstrip("/")
        self._retries = retries
        http2 = _http2_available() and transport is None
        self._client = httpx.AsyncClient(
            base_url=f"{self._base}{self._prefix}",
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            follow_redirects=True,
            http2=http2,
            transport=transport,
        )
        logger.info("BackendClient host=%s prefix=%s http2=%s retries=%d", httpx.URL(self._base).host, self._prefix, http2, retries)

    @property
    def base_url(self) -> str:
        return self._base

    async def close(self) -> None:
        await self._client.aclose()

    async def get_root(self, path: str, timeout: float = 10.0) -> httpx.Response:
        """GET a path at the service root (for example /health), outside the API prefix."""
        return await self._client.get(f"{self._base}{path}", timeout=timeout)

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
    ) -> Any:
        actual_path = path
        for key, value in (path_params or {}).items():
            actual_path = actual_path.replace(f"{{{key}}}", str(value))

        headers: dict[str, str] = {
            "Accept": "application/json",
            "X-Correlation-ID": correlation_id or uuid.uuid4().hex,
        }
        bare = _normalize_token(token)
        if bare:
            headers["Authorization"] = f"Bearer {bare}"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        is_get = method.upper() == "GET"
        attempts = self._retries + 1 if is_get else 1
        started = time.monotonic()
        response: httpx.Response | None = None
        for attempt in range(attempts):
            try:
                response = await self._client.request(
                    method=method.upper(), url=actual_path, params=params, json=json_body, headers=headers
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError) as exc:
                if attempt + 1 >= attempts:
                    logger.error("backend unreachable %s %s: %r", method, actual_path, exc)
                    raise ToolError(
                        "The marketplace service is not reachable right now.",
                        error_kind="backend_unreachable",
                    ) from exc
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            if response.status_code in (502, 503, 504) and attempt + 1 < attempts:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            break
        assert response is not None

        logger.info(
            "backend %s %s -> %s in %dms", method.upper(), actual_path, response.status_code, int((time.monotonic() - started) * 1000)
        )
        try:
            body = response.json() if response.content else {}
        except Exception:
            body = {"raw": response.text[:500]}

        if response.status_code >= 300:
            kind = _classify_error(response.status_code)
            detail = body.get("detail") if isinstance(body, dict) else None
            raise ToolError(
                message=detail if isinstance(detail, str) else f"Backend returned {response.status_code}",
                status_code=response.status_code,
                error_kind=kind,
                response_body=body if isinstance(body, dict) else {"body": body},
            )
        return body


def _classify_error(status_code: int) -> str:
    if status_code == 401:
        return "unauthenticated"
    if status_code == 403:
        return "permission_denied"
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        return "conflict"
    if status_code == 422:
        return "validation_error"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "backend_error"
    if 300 <= status_code < 400:
        return "unexpected_redirect"
    return "unknown"
