"""Share-link tools — create_share_link, resolve_share_link."""

from __future__ import annotations

from typing import Any, Optional

from ..backend_client import BackendClient
from ..idempotency import derive_idempotency_key


def register_links_tools(server: Any, client: BackendClient, session: Any) -> None:
    """Register share-link tools with the MCP server."""

    @server.tool(
        name="create_share_link",
        description=(
            "Create a shareable link for the merchant's store or a chosen set of "
            "products. Omit product_ids for a store link (visitors see all active "
            "products); pass product_ids for a product-set link (visitors see only "
            "those products). Merchant-only. Returns the public URL to share."
        ),
    )
    async def create_share_link(
        store_id: str,
        product_ids: Optional[list[str]] = None,
        title: Optional[str] = None,
        token: Optional[str] = None,
    ) -> dict:
        """Create a store or product-set share link."""
        body: dict[str, Any] = {"store_id": store_id}
        if product_ids:
            body["product_ids"] = product_ids
        if title:
            body["title"] = title
        idempotency_key = derive_idempotency_key(
            session.user_id, session.current_turn_id, "create_share_link", body,
        )
        return await client.call(
            "POST",
            "/links/",
            token=token,
            json_body=body,
            idempotency_key=idempotency_key,
        )

    @server.tool(
        name="resolve_share_link",
        description=(
            "Resolve a public share-link code (the part after /s/) to the store and "
            "the products it exposes. Read-only, works for anonymous visitors. Use "
            "this when a visitor opens a shared link so the agent can show exactly "
            "the products embedded in it."
        ),
    )
    async def resolve_share_link(
        code: str,
        token: Optional[str] = None,
    ) -> dict:
        """Resolve a share-link code publicly."""
        return await client.call(
            "GET",
            "/links/public/{code}",
            token=None,  # public endpoint — no auth needed
            path_params={"code": code},
        )
