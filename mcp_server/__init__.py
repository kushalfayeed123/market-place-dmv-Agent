"""MCP Server — thin, typed HTTP wrappers over the backend API.

Each tool is a direct, authenticated call to an existing backend endpoint.
The server holds and uses the user's backend access token (never the model).
"""

__all__ = ["create_server", "ToolError", "ConfirmationRequiredError"]

from .server import create_server
from .backend_client import ToolError
from .guards import ConfirmationRequiredError
