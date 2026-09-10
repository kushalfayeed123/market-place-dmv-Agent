"""Compatibility shim for mcp.server.Server.

mcp 1.10.0 (which works without pywin32 on Windows) uses separate
``list_tools`` and ``call_tool`` decorators.  Newer mcp versions
(2.x) provide a combined ``server.tool()`` decorator that registers
both in one step.

This shim adds a ``tool`` method to Server when it is missing, so
that the rest of the codebase can unconditionally write
``@server.tool(name=..., description=...)``.

No pywin32 / pywintypes is required – mcp 1.10.0 does not import
those modules at startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server import Server as MCPServer


def _tool_decorator(
    self,  # MCPServer
    /,
    name: str,
    description: str = "",
    *,
    inputSchema: dict[str, Any] | None = None,
):
    """Combined tool registration decorator for mcp < 2.x.

    Wraps ``self.list_tools()`` + ``self.call_tool()`` so callers can
    write ``@server.tool(name=..., description=...)`` uniformly.
    """

    def decorator(func):
        # Register the list-tools handler (provides tool metadata)
        @self.list_tools()
        async def _list():
            from mcp import types as mcp_types

            return [
                mcp_types.Tool(
                    name=name,
                    description=description,
                    inputSchema=inputSchema or {"type": "object", "properties": {}},
                )
            ]

        # Register the call-tool handler (executes the function)
        @self.call_tool()
        async def _call(tool_name: str, arguments: dict[str, Any]) -> Any:
            if tool_name != name:
                raise RuntimeError(f"Tool mismatch: expected {name}, got {tool_name}")
            return await func(tool_name, arguments)

        return func

    return decorator


def apply_shim() -> None:
    """Attach ``Server.tool`` if the installed mcp version lacks it."""
    try:
        from mcp.server import Server
    except ImportError:
        return

    if not hasattr(Server, "tool"):
        Server.tool = _tool_decorator  # type: ignore[attr-defined]


apply_shim()

