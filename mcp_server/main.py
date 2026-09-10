"""MCP Server entry point — run as a standalone process."""

from __future__ import annotations

import os

import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()


def main():
    """Run the MCP server."""
    from .backend_client import BackendClient
    from .guards import ConfirmationGate
    from .server import create_server

    config = {
        "backend_api_url": os.getenv("BACKEND_API_URL", "http://localhost:8000"),
        "backend_api_prefix": os.getenv("BACKEND_API_PREFIX", "/api/v1"),
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        "namespace": os.getenv("AGENT_REDIS_NAMESPACE", "agent"),
        "ttl": int(os.getenv("CONFIRMED_TOKEN_TTL", "300")),
        "host": os.getenv("MCP_SERVER_HOST", "0.0.0.0"),
        "port": int(os.getenv("MCP_SERVER_PORT", "8002")),
    }

    redis_client = redis.from_url(config["redis_url"], decode_responses=True)
    backend_client = BackendClient(
        base_url=config["backend_api_url"],
        api_prefix=config["backend_api_prefix"],
    )
    gate = ConfirmationGate(
        redis_client=redis_client,
        namespace=config["namespace"],
        ttl=config["ttl"],
    )

    # Create the server (tools are registered here)
    # Note: In a real deployment, the session would be per-request.
    # For the standalone server, we create a placeholder session.
    class PlaceholderSession:
        session_id = "standalone"
        user_id = None
        role = None
        turn_counter = 0
        def next_turn(self): self.turn_counter += 1; return self.turn_counter
        @property
        def current_turn_id(self): return self.turn_counter
        def get_cart(self): return {"items": []}
        def set_cart(self, cart): pass
        def clear_cart(self): pass

    server = create_server(backend_client, gate, PlaceholderSession())

    print(f"MCP Server starting on {config['host']}:{config['port']}")
    # In production, this would start the MCP server's own HTTP/SSE transport
    # For now, the server is used via the Gateway


if __name__ == "__main__":
    main()
