"""MCP Server — creates the FastMCP app and registers all tools.

The MCP server is the only thing that holds and uses the user's backend access
token. The model only ever sees tool names, schemas, and results.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

# ----------------------------------------------------------------------
# Windows pywin32 compatibility shim
# ----------------------------------------------------------------------
# The mcp package (>=1.10.0) imports mcp.os.win32.utilities at module
# import time on Windows, which in turn requires pywin32/pywintypes.
# In environments where pywin32 cannot be installed (no C compiler,
# no network for the 6.9 MB wheel), we inject stub modules so that
# `from mcp.server import Server` succeeds.
# ----------------------------------------------------------------------
if sys.platform == "win32":
    import types

    # Stub: mcp.os.win32.utilities
    _util = types.ModuleType("mcp.os.win32.utilities")
    _util.rebind_std_handle_to_fd = lambda fd: None
    _util._get_std_handle = lambda n: None
    _util._set_std_handle = lambda n, h: None
    sys.modules["mcp.os.win32.utilities"] = _util

    # Stub: mcp.os.win32 package
    _win32 = types.ModuleType("mcp.os.win32")
    sys.modules["mcp.os.win32"] = _win32

    # Stub: pywintypes (needed by win32com / mcp.os.win32)
    _pyw = types.ModuleType("pywintypes")
    _pyw.com_error = type("com_error", (Exception,), {})
    sys.modules["pywintypes"] = _pyw

    # Stub: win32con (needed by mcp.os.win32.utilities)
    _wc = types.ModuleType("win32con")
    _wc.STD_INPUT_HANDLE = -10
    _wc.STD_OUTPUT_HANDLE = -11
    _wc.STD_ERROR_HANDLE = -12
    _wc.DUPLICATE_SAME_ACCESS = 0x00000002
    _wc.GetStdHandle = lambda n: None
    _wc.DuplicateHandle = lambda *a, **k: None
    sys.modules["win32con"] = _wc

from .backend_client import BackendClient
from .guards import ConfirmationGate
from .tools import (
    register_admin_tools,
    register_cart_tools,
    register_catalog_tools,
    register_fulfillment_tools,
    register_knowledge_tools,
    register_ledger_tools,
    register_merchants_tools,
    register_orders_tools,
    register_payments_tools,
)


class ToolContext:
    """Per-call context injected into tool execution.

    This is the ONLY place the user's backend access token lives during a tool
    call. It is never part of the text the model sees.
    """

    def __init__(
        self,
        user_token: Optional[str],
        user_id: Optional[str],
        role: Optional[str],
        session_id: str,
        turn_id: int,
        confirmed_token: Optional[str] = None,
    ):
        self.user_token = user_token
        self.user_id = user_id
        self.role = role
        self.session_id = session_id
        self.turn_id = turn_id
        self.confirmed_token = confirmed_token


class SessionBridge:
    """Bridge between the Gateway session and the MCP tools.

    Provides the session cart operations and current turn info that tools need.
    """

    def __init__(self, session: Any):
        self._session = session

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def current_turn_id(self) -> int:
        return self._session.current_turn_id

    @property
    def user_id(self) -> Optional[str]:
        return self._session.user_id

    def get_cart(self) -> dict:
        return self._session.get_cart()

    def set_cart(self, cart: dict) -> None:
        self._session.set_cart(cart)

    def clear_cart(self) -> None:
        self._session.clear_cart()


def create_server(
    backend_client: BackendClient,
    gate: ConfirmationGate,
    session: Any,
    redis_client: Any = None,
    vector_namespace: str = "agent",
) -> Any:
    """Create and configure the MCP server with all tools registered.

    Args:
        backend_client: HTTP client for the backend API.
        gate: Confirmation gate for mutating_financial tools.
        session: The Gateway session (provides cart + turn info).

    Returns:
        The configured MCP server instance.
    """
    # Ensure mcp compatibility shim is applied before importing mcp classes.
    # mcp 1.10.0 (installed as a workaround for the pywin32 download issue
    # in this environment) lacks the ``@server.tool`` decorator that mcp 2.x
    # provides.  The shim adds it back using ``list_tools`` + ``call_tool``.
    from .mcp_compat import apply_shim as _apply_shim

    _apply_shim()

    # Import mcp here to avoid hard dependency at module level
    from mcp.server import Server

    server: Server = Server("marketplace-agent-mcp")
    bridge = SessionBridge(session)

    # Register all tool modules
    register_catalog_tools(server, backend_client)
    register_cart_tools(server, bridge)
    register_orders_tools(server, backend_client, gate, bridge)
    register_payments_tools(server, backend_client, gate, bridge)
    register_ledger_tools(server, backend_client)
    register_merchants_tools(server, backend_client, bridge)
    register_admin_tools(server, backend_client, gate, bridge)
    register_fulfillment_tools(server, backend_client, bridge)

    # Knowledge/RAG tools need Redis; register only when available.
    if redis_client is not None:
        register_knowledge_tools(server, redis_client, vector_namespace)

    return server
