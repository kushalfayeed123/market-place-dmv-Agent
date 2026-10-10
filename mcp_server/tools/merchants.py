"""Merchant tools — get_merchant_profile, list_merchants, create_product, update_inventory."""

from __future__ import annotations

from typing import Any, Optional

from ..backend_client import BackendClient
from ..idempotency import derive_idempotency_key


def register_merchants_tools(server: Any, client: BackendClient, session: Any) -> None:
    """Register merchant tools."""

    @server.tool(
        name="get_merchant_profile",
        description="Get merchant profile. Read-only.",
    )
    async def get_merchant_profile(
        merchant_id: str,
        token: Optional[str] = None,
    ) -> dict:
        """Get a merchant's profile."""
        return await client.call(
            "GET", "/merchants/{merchant_id}",
            token=token, path_params={"merchant_id": merchant_id},
        )

    @server.tool(
        name="list_merchants",
        description="List all merchants. Admin only. Read-only.",
    )
    async def list_merchants(
        skip: int = 0,
        limit: int = 50,
        token: Optional[str] = None,
    ) -> list[dict]:
        """List all merchants (admin)."""
        params = {"skip": skip, "limit": limit}
        body = await client.call("GET", "/merchants/", token=token, params=params)
        items = body.get("items", body) if isinstance(body, dict) else body
        if not isinstance(items, list):
            items = [items] if isinstance(items, dict) else []
        return items

    @server.tool(
        name="create_product",
        description="Create a new product. Medium risk — mutating, non-financial.",
    )
    async def create_product(
        title: str,
        description: Optional[str] = None,
        base_price_amount: int = 0,
        base_price_currency: str = "NGN",
        fulfillment_type: str = "physical",
        category_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        store_id: Optional[str] = None,
        slug: Optional[str] = None,
        urls: Optional[list[str]] = None,
        status: Optional[str] = None,
        token: Optional[str] = None,
    ) -> dict:
        """Create a new product.

        ``merchant_id`` and ``store_id`` are optional — when omitted the backend
        derives them from the authenticated user's merchant and primary store.
        ``slug`` is auto-generated from ``title`` when not supplied.
        """
        body_data: dict[str, Any] = {
            "title": title,
            "description": description,
            "base_price_amount": base_price_amount,
            "base_price_currency": base_price_currency,
            "fulfillment_type": fulfillment_type,
        }
        if category_id:
            body_data["category_id"] = category_id
        if merchant_id:
            body_data["merchant_id"] = merchant_id
        if store_id:
            body_data["store_id"] = store_id
        if slug:
            body_data["slug"] = slug
        if urls:
            body_data["urls"] = urls
        if status:
            body_data["status"] = status

        idempotency_key = derive_idempotency_key(
            session.user_id, session.current_turn_id, "create_product", body_data,
        )
        return await client.call(
            "POST", "/catalog/products",
            token=token,
            json_body=body_data,
            idempotency_key=idempotency_key,
        )

    @server.tool(
        name="update_inventory",
        description="Update inventory for a variant. Medium risk — mutating, non-financial.",
    )
    async def update_inventory(
        variant_id: str,
        quantity_available: int,
        token: Optional[str] = None,
    ) -> dict:
        """Update inventory quantity for a variant."""
        body_data = {"quantity_available": quantity_available}
        idempotency_key = derive_idempotency_key(
            session.user_id, session.current_turn_id, "update_inventory", body_data,
        )
        return await client.call(
            "PUT", "/catalog/inventory/{variant_id}",
            token=token,
            json_body=body_data,
            path_params={"variant_id": variant_id},
            idempotency_key=idempotency_key,
        )

    @server.tool(
        name="update_product",
        description="Partially update a product (PATCH). Merchant owner or platform admin only.",
    )
    async def update_product(
        product_id: str,
        title: Optional[str] = None,
        slug: Optional[str] = None,
        description: Optional[str] = None,
        fulfillment_type: Optional[str] = None,
        base_price_amount: Optional[int] = None,
        base_price_currency: Optional[str] = None,
        category_id: Optional[str] = None,
        status: Optional[str] = None,
        urls: Optional[list[str]] = None,
        token: Optional[str] = None,
    ) -> dict:
        """Partially update a product.

        Only supplied (non-None) fields are applied.
        ``base_price_amount`` is expected in the major unit representation
        (e.g. NGN 100.00) and will be sent as-is; the backend stores in
        minor units internally.
        """
        body_data: dict[str, Any] = {}
        if title is not None:
            body_data["title"] = title
        if slug is not None:
            body_data["slug"] = slug
        if description is not None:
            body_data["description"] = description
        if fulfillment_type is not None:
            body_data["fulfillment_type"] = fulfillment_type
        if base_price_amount is not None:
            body_data["base_price_amount"] = base_price_amount
        if base_price_currency is not None:
            body_data["base_price_currency"] = base_price_currency
        if category_id is not None:
            body_data["category_id"] = category_id
        if status is not None:
            body_data["status"] = status
        if urls is not None:
            body_data["urls"] = urls

        idempotency_key = derive_idempotency_key(
            session.user_id, session.current_turn_id, "update_product", body_data,
        )
        return await client.call(
            "PATCH", "/catalog/products/{product_id}",
            token=token,
            json_body=body_data,
            path_params={"product_id": product_id},
            idempotency_key=idempotency_key,
        )

    @server.tool(
        name="like_product",
        description="Like a product. Authenticated users only. Idempotent.",
    )
    async def like_product(
        product_id: str,
        token: Optional[str] = None,
    ) -> dict:
        """Like a product (idempotent — returns existing if already liked)."""
        return await client.call(
            "POST", "/catalog/products/{product_id}/like",
            token=token,
            path_params={"product_id": product_id},
        )

    @server.tool(
        name="unlike_product",
        description="Remove a like from a product. Authenticated users only.",
    )
    async def unlike_product(
        product_id: str,
        token: Optional[str] = None,
    ) -> dict:
        """Remove a like from a product."""
        return await client.call(
            "DELETE", "/catalog/products/{product_id}/like",
            token=token,
            path_params={"product_id": product_id},
        )

    @server.tool(
        name="get_product_likes",
        description="Get like count and current-user liked status for a product. Read-only.",
    )
    async def get_product_likes(
        product_id: str,
        token: Optional[str] = None,
    ) -> dict:
        """Get like count and whether the authenticated user has liked the product."""
        return await client.call(
            "GET", "/catalog/products/{product_id}/likes",
            token=token,
            path_params={"product_id": product_id},
        )
