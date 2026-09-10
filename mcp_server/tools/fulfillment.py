"""Fulfillment tools — get_fulfillment_status, list_fulfillments, create_fulfillment, update_fulfillment_status."""

from __future__ import annotations

from typing import Any

from mcp_server.backend_client import BackendClient
from mcp_server.idempotency import derive_idempotency_key


def register_fulfillment_tools(
    server: Any, client: BackendClient, session: Any
) -> None:
    """Register fulfillment tools."""

    @server.tool(
        name="get_fulfillment_status",
        description="Get fulfillment status for an order. Read-only.",
    )
    async def get_fulfillment_status(
        order_id: str,
        token: str | None = None,
    ) -> dict:
        """Get fulfillment status for an order."""
        return await client.call(
            "GET",
            "/fulfillment/order/{order_id}",
            token=token,
            path_params={"order_id": order_id},
        )

    @server.tool(
        name="list_fulfillments",
        description="List fulfillments for a merchant. Read-only.",
    )
    async def list_fulfillments(
        merchant_id: str | None = None,
        status: str | None = None,
        skip: int = 0,
        limit: int = 50,
        token: str | None = None,
    ) -> list[dict]:
        """List fulfillments, optionally filtered."""
        params: dict[str, Any] = {"skip": skip, "limit": limit}
        if merchant_id:
            params["merchant_id"] = merchant_id
        if status:
            params["status"] = status
        body = await client.call("GET", "/fulfillment/", token=token, params=params)
        items = body.get("items", body) if isinstance(body, dict) else body
        if not isinstance(items, list):
            items = [items] if isinstance(items, dict) else []
        return items

    @server.tool(
        name="create_fulfillment",
        description="Create a fulfillment record. Medium risk — mutating, non-financial.",
    )
    async def create_fulfillment(
        order_id: str,
        tracking_number: str | None = None,
        carrier: str | None = None,
        token: str | None = None,
    ) -> dict:
        """Create a fulfillment record."""
        body_data: dict[str, Any] = {"order_id": order_id}
        if tracking_number:
            body_data["tracking_number"] = tracking_number
        if carrier:
            body_data["carrier"] = carrier

        idempotency_key = derive_idempotency_key(
            session.user_id,
            session.current_turn_id,
            "create_fulfillment",
            body_data,
        )
        return await client.call(
            "POST",
            "/fulfillment/",
            token=token,
            json_body=body_data,
            idempotency_key=idempotency_key,
        )

    @server.tool(
        name="update_fulfillment_status",
        description="Update fulfillment status. Medium risk — mutating, non-financial.",
    )
    async def update_fulfillment_status(
        fulfillment_id: str,
        status: str,
        tracking_number: str | None = None,
        token: str | None = None,
    ) -> dict:
        """Update fulfillment status."""
        body_data: dict[str, Any] = {"status": status}
        if tracking_number:
            body_data["tracking_number"] = tracking_number

        idempotency_key = derive_idempotency_key(
            session.user_id,
            session.current_turn_id,
            "update_fulfillment_status",
            body_data,
        )
        return await client.call(
            "PUT",
            "/fulfillment/{fulfillment_id}/status",
            token=token,
            json_body=body_data,
            path_params={"fulfillment_id": fulfillment_id},
            idempotency_key=idempotency_key,
        )
