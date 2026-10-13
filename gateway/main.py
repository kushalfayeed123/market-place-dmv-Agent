"""Agent Gateway: FastAPI app with SSE streaming, health checks and guarded debug endpoints."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Re-exported for backwards compatibility (tests and other modules import them from here).
from .streaming import coalesce_text, finalize_directives, sse, stream_turn  # noqa: F401
from .tokens import normalize_token

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("gateway")

_MERCHANT_ROLES = {"merchant_owner", "merchant_staff", "platform_admin"}


class SseRequest(BaseModel):
    """Request body for the /sse endpoint."""

    message: str = Field(..., description="The user's message to the agent")
    session_id: str | None = Field(None, description="Existing session ID to resume a conversation.")
    confirmed_token: str | None = Field(None, description="Token from a prior user_confirmed directive.")
    user_token: str | None = Field(
        None,
        description="The signed-in user's backend access token. 'Bearer <token>' and '<token>' are both accepted.",
    )


# ----------------------------------------------------------------------------- config


def _load_config() -> dict:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    if redis_url.startswith("https://"):
        redis_url = "rediss://" + redis_url[len("https://"):]
    elif redis_url.startswith("http://"):
        redis_url = "redis://" + redis_url[len("http://"):]
    elif not redis_url.startswith(("redis://", "rediss://", "unix://")):
        redis_url = f"redis://{redis_url}"
    return {
        "backend_api_url": os.getenv("BACKEND_API_URL", "http://localhost:8000"),
        "backend_api_prefix": os.getenv("BACKEND_API_PREFIX", "/api/v1"),
        "redis_url": redis_url,
        "agent_namespace": os.getenv("AGENT_REDIS_NAMESPACE", "agent"),
        "model_provider": os.getenv("AGENT_MODEL_PROVIDER", "nvidia"),
        "model_name": os.getenv("AGENT_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct"),
        "nvidia_api_key": os.getenv("NVIDIA_API_KEY", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        # Debug endpoints are OFF unless AGENT_DEBUG=1, and then require X-Debug-Key.
        "debug_enabled": os.getenv("AGENT_DEBUG", "0") == "1",
        "debug_key": os.getenv("AGENT_DEBUG_KEY", ""),
        "gateway_host": os.getenv("AGENT_GATEWAY_HOST", "0.0.0.0"),
        "gateway_port": int(os.getenv("AGENT_GATEWAY_PORT", os.getenv("PORT", "8001"))),
        "confirmed_token_ttl": int(os.getenv("CONFIRMED_TOKEN_TTL", "300")),
        "turn_timeout": float(os.getenv("AGENT_TURN_TIMEOUT", "100")),
        "heartbeat": float(os.getenv("AGENT_HEARTBEAT", "15")),
        "kb_sync_enabled": os.getenv("KB_SYNC_ENABLED", "1") == "1",
    }


def _is_production() -> bool:
    return bool(os.getenv("RENDER")) or os.getenv("ENV", "").lower() == "production"


def validate_config(config: dict) -> None:
    """Fail loudly at startup instead of silently searching an unreachable localhost."""
    if not _is_production():
        return
    problems = []
    backend = config["backend_api_url"]
    if not os.getenv("BACKEND_API_URL") or "localhost" in backend or "127.0.0.1" in backend:
        problems.append(f"BACKEND_API_URL is missing or points at localhost ({backend})")
    if "localhost" in config["redis_url"] or "127.0.0.1" in config["redis_url"]:
        problems.append("REDIS_URL points at localhost")
    if config["model_provider"] == "nvidia" and not config["nvidia_api_key"]:
        problems.append("NVIDIA_API_KEY is not set")
    if config["debug_enabled"] and not config["debug_key"]:
        problems.append("AGENT_DEBUG=1 requires AGENT_DEBUG_KEY")
    if problems:
        raise RuntimeError("Invalid production configuration: " + "; ".join(problems))


def _create_redis(url: str) -> redis.Redis:
    kwargs: dict[str, Any] = {"decode_responses": True}
    if url.startswith("rediss://"):
        kwargs["ssl_cert_reqs"] = None
    return redis.from_url(url, **kwargs)


def _load_system_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "system.md")
    try:
        with open(prompt_path, encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, UnicodeDecodeError):
        logger.warning("system prompt not found at %s; using a minimal default", prompt_path)
        return "You are a helpful marketplace assistant."


# ----------------------------------------------------------------------------- background tasks


async def _connect_redis(app: FastAPI) -> None:
    delay, max_delay = 1.0, 30.0
    while True:
        try:
            await app.state.redis.ping()
            app.state.redis_connected = True
            logger.info("Redis connected")
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            app.state.redis_connected = False
            logger.warning("Redis connection failed: %s. Retrying in %.0fs", exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)


async def _kb_sync_loop(app: FastAPI) -> None:
    """Keep the knowledge index fresh: one full sync at startup, then incremental syncs."""
    from mcp_server.knowledge.config import load_knowledge_config
    from mcp_server.knowledge.sync import run_sync

    status = app.state.kb_status
    cfg = load_knowledge_config()
    while not app.state.redis_connected:
        await asyncio.sleep(1)
    first = True
    try:
        while True:
            started = time.monotonic()
            try:
                result = await run_sync(app.state.redis, cfg, full=first)
                status.update(last_sync_at=time.time(), last_error=None, last_result=repr(result)[:200], mode="full" if first else "incremental")
                logger.info("KB sync ok (%s) in %.1fs: %s", status["mode"], time.monotonic() - started, status["last_result"])
                first = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                status["last_error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("KB sync failed")
            status["runs"] += 1
            await asyncio.sleep(cfg.sync_interval_seconds)
    except asyncio.CancelledError:
        return


async def _self_test(app: FastAPI) -> None:
    """Log, at startup, whether the backend is reachable and returns products."""
    result: dict[str, Any] = {"status": "running"}
    app.state.selftest = result
    try:
        client = app.state.get_backend_client()
        health = await client.get_root("/health")
        result["backend_health"] = health.status_code
        body = await client.call("GET", "/catalog/products", params={"limit": 1})
        items = body.get("items", body) if isinstance(body, dict) else body
        result["catalog_items_returned"] = len(items) if isinstance(items, list) else -1
        result["status"] = "ok"
        logger.info("startup self-test ok: %s", result)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        logger.error("STARTUP SELF-TEST FAILED: the agent cannot reach the catalog (%s)", result["error"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.state.config
    # Initialise ALL state before any task starts.
    app.state.redis = _create_redis(config["redis_url"])
    app.state.redis_connected = False
    app.state.backend_client = None
    app.state.gate = None
    app.state.model_backend = None
    app.state.debug_tracer = None
    app.state.agent = None
    app.state.knowledge_service = None
    app.state.kb_status = {"runs": 0, "last_sync_at": None, "last_error": None}
    app.state.selftest = {"status": "pending"}

    tasks = [asyncio.create_task(_connect_redis(app)), asyncio.create_task(_self_test(app))]
    if config["kb_sync_enabled"]:
        app.state.kb_sync_task = asyncio.create_task(_kb_sync_loop(app))
        tasks.append(app.state.kb_sync_task)
    yield

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if app.state.backend_client is not None:
        await app.state.backend_client.close()
    closer = getattr(app.state.redis, "aclose", None) or app.state.redis.close
    await closer()


# ----------------------------------------------------------------------------- app


def create_app() -> FastAPI:
    config = _load_config()
    validate_config(config)
    app = FastAPI(title="Marketplace Agent Gateway", version="0.2.0", lifespan=lifespan)
    app.state.config = config

    origins = [
        o.strip()
        for o in os.getenv(
            "AGENT_CORS_ORIGINS",
            "http://localhost:8443,http://localhost:3000,https://market-place-dmv-fe.vercel.app",
        ).split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=os.getenv("AGENT_CORS_ORIGIN_REGEX") or None,  # e.g. https://.*\.vercel\.app for previews
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_backend_client():
        if app.state.backend_client is None:
            from mcp_server.backend_client import BackendClient

            app.state.backend_client = BackendClient(
                base_url=config["backend_api_url"], api_prefix=config["backend_api_prefix"]
            )
        return app.state.backend_client

    app.state.get_backend_client = get_backend_client

    def get_gate():
        if app.state.gate is None:
            from mcp_server.guards import ConfirmationGate

            app.state.gate = ConfirmationGate(
                redis_client=app.state.redis, namespace=config["agent_namespace"], ttl=config["confirmed_token_ttl"]
            )
        return app.state.gate

    def get_debug_tracer():
        if app.state.debug_tracer is None:
            from .debug import DebugTracer

            app.state.debug_tracer = DebugTracer(
                redis_client=app.state.redis, namespace=config["agent_namespace"], enabled=config["debug_enabled"]
            )
        return app.state.debug_tracer

    def get_model_backend():
        if app.state.model_backend is None:
            provider = config["model_provider"]
            api_key = config["nvidia_api_key"] if provider == "nvidia" else config["anthropic_api_key"]
            if provider == "nvidia":
                from .model_router import NIM_MODELS, FailoverBackend

                app.state.model_backend = FailoverBackend(models=NIM_MODELS, api_key=api_key)
                logger.info("FailoverBackend with %d models, primary %s", len(NIM_MODELS), NIM_MODELS[0].name)
            else:
                from .provider import get_backend

                app.state.model_backend = get_backend(provider, config["model_name"], api_key)
        return app.state.model_backend

    def get_knowledge_service():
        if app.state.knowledge_service is not None:
            return app.state.knowledge_service
        from mcp_server.knowledge.config import load_knowledge_config
        from mcp_server.knowledge.embedders import create_embedder
        from mcp_server.knowledge.service import KnowledgeService

        kb_config = load_knowledge_config()
        embedder = create_embedder(
            provider=kb_config.embedding_provider,
            api_key=kb_config.nvidia_api_key,
            model=kb_config.embedding_model,
            dim=kb_config.embedding_dim,
            batch_size=kb_config.embedding_batch_size,
            timeout=kb_config.embedding_timeout,
        )
        app.state.knowledge_service = KnowledgeService(redis_client=app.state.redis, embedder=embedder, config=kb_config)
        return app.state.knowledge_service

    def get_agent():
        if app.state.agent is None:
            from .agent import Agent

            try:
                knowledge = get_knowledge_service()
            except Exception:
                # Products are answered from the live catalog; a broken index must not take search down.
                logger.exception("knowledge service unavailable; continuing without it")
                knowledge = None
            app.state.agent = Agent(
                backend_client=get_backend_client(),
                gate=get_gate(),
                model_backend=get_model_backend(),
                debug_tracer=get_debug_tracer(),
                system_prompt=_load_system_prompt(),
                redis_client=app.state.redis,
                vector_namespace=config["agent_namespace"],
                knowledge_service=knowledge,
            )
        return app.state.agent

    def get_session_store():
        from .session import SessionStore

        return SessionStore(redis_client=app.state.redis, namespace=config["agent_namespace"])

    def require_debug(key: str | None) -> None:
        if not config["debug_enabled"] or not config["debug_key"]:
            raise HTTPException(404, "Not found")
        if not key or not hmac.compare_digest(key, config["debug_key"]):
            raise HTTPException(403, "Invalid debug key")

    async def resolve_identity(session) -> None:
        """Fill user_id and role from the backend; drop an expired or invalid token."""
        if not session.user_token:
            session.user_id = session.role = None
            return
        try:
            body = await asyncio.wait_for(
                get_backend_client().call("GET", "/auth/me", token=session.user_token), timeout=6.0
            )
            if isinstance(body, dict):
                session.user_id = str(body.get("id") or "") or None
                session.role = body.get("role") or None
        except Exception as exc:
            kind = getattr(exc, "error_kind", "")
            if kind in ("unauthenticated", "permission_denied"):
                logger.info("dropping an expired or invalid user token for session %s", session.session_id)
                session.user_token = None
                session.user_id = session.role = None
            else:
                logger.warning("could not resolve identity: %r", exc)

    # ---------------------------------------------------------------- health

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "agent-gateway", "redis": "connected" if app.state.redis_connected else "disconnected"}

    @app.get("/ready")
    async def ready():
        if not app.state.redis_connected:
            raise HTTPException(503, "Redis unavailable")
        if app.state.selftest.get("status") == "failed":
            raise HTTPException(503, f"Backend unreachable: {app.state.selftest.get('error')}")
        return {"ready": True, "redis": "ok", "backend": app.state.selftest.get("status")}

    # ---------------------------------------------------------------- chat

    @app.post("/sse")
    async def sse_stream(body: SseRequest):
        """Stream one agent turn to the browser via SSE."""
        user_token = normalize_token(body.user_token)  # accept 'Bearer x' and 'x'

        async def failed(message: str) -> AsyncIterator[str]:
            yield sse("error", {"error": message})

        try:
            sessions = get_session_store()
            session = await sessions.get_session(body.session_id) if body.session_id else None
            if session is None:
                session = await sessions.create_session(user_token=user_token)
                await resolve_identity(session)
            elif session.user_token != user_token:  # newest token wins; a missing token means signed out
                session.user_token = user_token
                await resolve_identity(session)
            agent = get_agent()
        except Exception:
            logger.exception("could not start a turn")
            return StreamingResponse(
                failed("The assistant is temporarily unavailable. Please try again in a moment."),
                media_type="text/event-stream",
            )

        return StreamingResponse(
            stream_turn(
                agent=agent,
                session=session,
                message=body.message,
                confirmed_token=body.confirmed_token,
                save_session=lambda: sessions.save_session(session),
                turn_timeout=config["turn_timeout"],
                heartbeat=config["heartbeat"],
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    # ---------------------------------------------------------------- debug (guarded)

    @app.get("/debug/config")
    async def debug_config(x_debug_key: str | None = Header(None)):
        require_debug(x_debug_key)
        return {
            "backend_host": __import__("urllib.parse", fromlist=["urlparse"]).urlparse(config["backend_api_url"]).netloc,
            "backend_prefix": config["backend_api_prefix"],
            "redis_connected": app.state.redis_connected,
            "model_provider": config["model_provider"],
            "selftest": app.state.selftest,
            "kb": app.state.kb_status,
        }

    @app.get("/debug/search")
    async def debug_search(q: str = Query(...), x_debug_key: str | None = Header(None)):
        """Run the exact product search pipeline and show each step. The fastest way to see why a search is empty."""
        require_debug(x_debug_key)
        from .search_pipeline import search_catalog

        client = get_backend_client()
        calls: list[dict[str, Any]] = []

        async def fetch(params: dict[str, Any]):
            body = await client.call("GET", "/catalog/products", params=params)
            items = body.get("items", body) if isinstance(body, dict) else body
            calls.append({"params": params, "returned": len(items) if isinstance(items, list) else -1})
            return items if isinstance(items, list) else []

        outcome = await search_catalog(fetch, query=q, limit=10)
        return {
            "strategy": outcome.strategy,
            "total": outcome.total,
            "steps": outcome.attempts,
            "backend_calls": calls,
            "sample": [{"id": p.get("id"), "title": p.get("title") or p.get("name"), "status": p.get("status")} for p in outcome.items],
        }

    @app.get("/debug/sessions/{session_id}/trace")
    async def debug_trace(session_id: str, x_debug_key: str | None = Header(None)):
        require_debug(x_debug_key)
        tracer = get_debug_tracer()
        trace = tracer.get_trace(session_id) or await tracer.load_trace(session_id)
        return {"session_id": session_id, "events": trace}

    @app.get("/debug/sessions/{session_id}/events")
    async def debug_events(session_id: str, x_debug_key: str | None = Header(None)):
        require_debug(x_debug_key)
        tracer = get_debug_tracer()

        async def event_stream() -> AsyncIterator[str]:
            last_count = 0
            while True:
                events = tracer.get_trace(session_id)
                for event in events[last_count:]:
                    yield f"event: {event['event_type']}\ndata: {json.dumps(event)}\n\n"
                last_count = len(events)
                await asyncio.sleep(0.5)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


if __name__ == "__main__":
    import uvicorn

    cfg = _load_config()
    uvicorn.run(create_app(), host=cfg["gateway_host"], port=cfg["gateway_port"])
