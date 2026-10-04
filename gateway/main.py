"""Agent Gateway — FastAPI app with SSE streaming and debug endpoints."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()


class SseRequest(BaseModel):
    """Request body for the /sse endpoint."""
    message: str = Field(..., description="The user's message to the agent")
    session_id: str | None = Field(None, description="Existing session ID to resume a conversation. Creates new session if omitted.")
    confirmed_token: str | None = Field(None, description="Token from a prior user_confirmed directive, used to authorize a destructive action.")


def _load_config() -> dict:
    """Load configuration from environment."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # Ensure Redis URL has proper scheme — handle http(s):// URLs from Upstash console
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
        "debug_enabled": os.getenv("AGENT_DEBUG", "1") == "1",
        "gateway_host": os.getenv("AGENT_GATEWAY_HOST", "0.0.0.0"),
        "gateway_port": int(os.getenv("AGENT_GATEWAY_PORT", "8001")),
        "confirmed_token_ttl": int(os.getenv("CONFIRMED_TOKEN_TTL", "300")),
    }


def _create_redis(url: str) -> redis.Redis:
    kwargs = {"decode_responses": True}
    if url.startswith("rediss://"):
        kwargs["ssl_cert_reqs"] = None
    return redis.from_url(url, **kwargs)


def _load_system_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "system.md")
    try:
        with open(prompt_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful marketplace assistant."


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.state.config
    app.state.redis = _create_redis(config["redis_url"])
    app.state.redis_connected = False
    
  
    # Start background task to connect to Redis (don't block startup)
    redis_connected = asyncio.Event()

    async def _connect_redis():
        """Background task to establish Redis connection with retries."""
        retry_delay = 1.0
        max_retry_delay = 30.0
        while True:
            try:
                await app.state.redis.ping()
                app.state.redis_connected = True
                redis_connected.set()
                print(f"[GATEWAY] Redis connected successfully")
                return
            except Exception as exc:
                app.state.redis_connected = False
                print(f"[GATEWAY] Redis connection failed: {exc}. Retrying in {retry_delay:.0f}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

    # Start Redis connection in background
    redis_task = asyncio.create_task(_connect_redis())

    app.state.backend_client = None
    app.state.gate = None
    app.state.model_backend = None
    app.state.debug_tracer = None
    app.state.agent = None
    yield
    redis_task.cancel()
    try:
        await redis_task
    except asyncio.CancelledError:
        pass
    await app.state.redis.close()



def create_app() -> FastAPI:
    """Create the FastAPI application."""
    config = _load_config()
    app = FastAPI(title="Marketplace Agent Gateway", version="0.1.0", lifespan=lifespan)
    app.state.config = config

    # ── CORS ──────────────────────────────────────────────
    # Allow the frontend dev origin (http://localhost:8443) and any origins
    # listed in AGENT_CORS_ORIGINS (comma-separated). With
    # allow_credentials=True the "*" wildcard is not permitted by the CORS
    # spec, so origins are enumerated explicitly.
    _cors_origins = [
        origin.strip()
        for origin in os.getenv(
            "AGENT_CORS_ORIGINS", "http://localhost:8443,http://localhost:3000"
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_backend_client():
        if app.state.backend_client is None:
            from mcp_server.backend_client import BackendClient
            app.state.backend_client = BackendClient(
                base_url=config["backend_api_url"],
                api_prefix=config["backend_api_prefix"],
            )
        return app.state.backend_client

    def get_gate():
        if app.state.gate is None:
            from mcp_server.guards import ConfirmationGate
            app.state.gate = ConfirmationGate(
                redis_client=app.state.redis,
                namespace=config["agent_namespace"],
                ttl=config["confirmed_token_ttl"],
            )
        return app.state.gate

    def get_debug_tracer():
        if app.state.debug_tracer is None:
            from .debug import DebugTracer
            app.state.debug_tracer = DebugTracer(
                redis_client=app.state.redis,
                namespace=config["agent_namespace"],
                enabled=config["debug_enabled"],
            )
        return app.state.debug_tracer

    def get_model_backend():
        if app.state.model_backend is None:
            provider = config["model_provider"]
            api_key = config["nvidia_api_key"] if provider == "nvidia" else config["anthropic_api_key"]

            # Use FailoverBackend for NVIDIA to enable model switching
            if provider == "nvidia":
                from .model_router import FailoverBackend, NIM_MODELS
                # Allow custom model list via env var, fallback to default NIM_MODELS
                app.state.model_backend = FailoverBackend(
                    models=NIM_MODELS,
                    api_key=api_key,
                )
                print(f"[GATEWAY] Initialized FailoverBackend with {len(NIM_MODELS)} models")
                print(f"[GATEWAY] Primary model: {NIM_MODELS[0].name}")
            else:
                from .provider import get_backend
                model = config["model_name"]
                app.state.model_backend = get_backend(provider, model, api_key)
        return app.state.model_backend

    def get_agent():
        if app.state.agent is None:
            from .agent import Agent
            system_prompt = _load_system_prompt()
            app.state.agent = Agent(
                backend_client=get_backend_client(),
                gate=get_gate(),
                model_backend=get_model_backend(),
                debug_tracer=get_debug_tracer(),
                system_prompt=system_prompt,
            )
        return app.state.agent

    def get_session_store():
        from .session import SessionStore
        return SessionStore(redis_client=app.state.redis, namespace=config["agent_namespace"])

    @app.get("/health")
    async def health():
        redis_status = "connected" if app.state.redis_connected else "disconnected"
        return {"status": "ok", "service": "agent-gateway", "redis": redis_status}

    @app.get("/ready")
    async def ready():
        """Readiness probe — verifies Redis connectivity for orchestrators."""
        if not app.state.redis_connected:
            raise HTTPException(503, "Redis unavailable")
        return {"ready": True, "redis": "ok"}

    @app.post("/sse")
    async def sse_stream(body: SseRequest):
        """Stream agent events to the browser via SSE."""
        session_id = body.session_id
        message = body.message
        confirmed_token = body.confirmed_token

        sessions = get_session_store()
        if session_id:
            session = await sessions.get_session(session_id)
            if not session:
                session = await sessions.create_session()
        else:
            session = await sessions.create_session()

        agent = get_agent()

        async def event_stream() -> AsyncIterator[str]:
            yield f"event: session\ndata: {json.dumps({'session_id': session.session_id})}\n\n"
            try:
                async for event in agent.run_turn(session, message, confirmed_token):
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                await sessions.save_session(session)
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/debug/sessions/{session_id}/trace")
    async def debug_trace(session_id: str):
        """Get the full debug trace for a session (postmortem)."""
        if not config["debug_enabled"]:
            raise HTTPException(403, "Debug mode is disabled")
        tracer = get_debug_tracer()
        trace = tracer.get_trace(session_id)
        if not trace:
            trace = await tracer.load_trace(session_id)
        return {"session_id": session_id, "events": trace}

    @app.get("/debug/sessions/{session_id}/events")
    async def debug_events(session_id: str):
        """Stream live debug events for a session."""
        if not config["debug_enabled"]:
            raise HTTPException(403, "Debug mode is disabled")
        tracer = get_debug_tracer()

        async def event_stream() -> AsyncIterator[str]:
            import asyncio
            last_count = 0
            while True:
                events = tracer.get_trace(session_id)
                new_events = events[last_count:]
                for event in new_events:
                    yield f"event: {event['event_type']}\ndata: {json.dumps(event)}\n\n"
                last_count = len(events)
                await asyncio.sleep(0.5)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


if __name__ == "__main__":
    import uvicorn
    config = _load_config()
    app = create_app()
    uvicorn.run(app, host=config["gateway_host"], port=config["gateway_port"])