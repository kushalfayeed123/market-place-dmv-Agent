"""Catalog tools — search_products, get_product_detail, get_categories."""

from __future__ import annotations

from typing import Any

from mcp_server.backend_client import BackendClient
from mcp_server.utils.price import (
    _map_price,
)
from schemas.catalog import (
    CategoryView,
    ProductCardView,
    ProductDetailView,
    VariantView,
)


def _map_variant(v: dict) -> VariantView:
    """Map a backend variant dict to VariantView."""
    return VariantView(
        id=str(v.get("id", "")),
        sku=v.get("sku", ""),
        name=v.get("name", ""),
        price=_map_price(
            v.get("price_override_amount") or v.get("base_price_amount"),
            v.get("price_override_currency") or v.get("base_price_currency"),
        ),
        quantity_available=v.get("quantity_available"),
        attributes=v.get("attributes", {}) or {},
    )


def _map_product_card(p: dict) -> ProductCardView:
    """Map a backend product list item to ProductCardView."""
    return ProductCardView(
        id=str(p.get("id", "")),
        name=p.get("name", ""),
        merchant_id=str(p.get("merchant_id", "")),
        merchant_name=p.get("merchant_name", ""),
        category_id=str(p.get("category_id", "") or ""),
        category_name=p.get("category_name"),
        price=_map_price(p.get("base_price_amount"), p.get("base_price_currency")),
        image_url=p.get("image_url"),
        quantity_available=p.get("quantity_available"),
        status=p.get("status", "active"),
    )


def _map_product_detail(p: dict) -> ProductDetailView:
    """Map a backend product detail response to ProductDetailView."""
    variants = [_map_variant(v) for v in (p.get("variants") or [])]
    return ProductDetailView(
        id=str(p.get("id", "")),
        name=p.get("name", ""),
        merchant_id=str(p.get("merchant_id", "")),
        merchant_name=p.get("merchant_name", ""),
        category_id=str(p.get("category_id", "") or ""),
        category_name=p.get("category_name"),
        price=_map_price(p.get("base_price_amount"), p.get("base_price_currency")),
        image_url=p.get("image_url"),
        quantity_available=p.get("quantity_available"),
        status=p.get("status", "active"),
        description=p.get("description"),
        variants=variants,
        attributes=p.get("attributes", {}) or {},
        created_at=p.get("created_at"),
        updated_at=p.get("updated_at"),
    )


def _map_category(c: dict) -> CategoryView:
    """Map a backend category response to CategoryView."""
    return CategoryView(
        id=str(c.get("id", "")),
        name=c.get("name", ""),
        slug=c.get("slug", ""),
        parent_id=str(c.get("parent_id", "") or "") or None,
        product_count=c.get("product_count"),
    )


def register_catalog_tools(server: Any, client: BackendClient) -> None:
    """Register catalog tools with the MCP server."""

    @server.tool(
        name="search_products",
        description="Search the product catalog. Read-only, safe, no confirmation needed.",
    )
    async def search_products(
        q: str | None = None,
        category_id: str | None = None,
        merchant_id: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
        currency: str | None = None,
        skip: int = 0,
        limit: int = 20,
        token: str | None = None,
    ) -> list[dict]:
        """Search products with optional filters."""
        params: dict[str, Any] = {"skip": skip, "limit": limit}
        if q:
            params["q"] = q
        if category_id:
            params["category_id"] = category_id
        if merchant_id:
            params["merchant_id"] = merchant_id
        if price_min is not None:
            params["price_min"] = price_min
        if price_max is not None:
            params["price_max"] = price_max
        if currency:
            params["currency"] = currency

        body = await client.call("GET", "/catalog/products", token=token, params=params)
        items = body.get("items", body) if isinstance(body, dict) else body
        if not isinstance(items, list):
            items = [items] if isinstance(items, dict) else []
        return [_map_product_card(p).model_dump() for p in items]

    @server.tool(
        name="get_product_detail",
        description="Get full product detail including variants. Read-only.",
    )
    async def get_product_detail(
        product_id: str,
        token: str | None = None,
    ) -> dict:
        """Get a single product by ID."""
        body = await client.call(
            "GET",
            "/catalog/products/{product_id}",
            token=token,
            path_params={"product_id": product_id},
        )
        return _map_product_detail(body).model_dump()

    @server.tool(
        name="get_categories",
        description="List product categories. Read-only.",
    )
    async def get_categories(
        token: str | None = None,
    ) -> list[dict]:
        """List all product categories."""
        body = await client.call("GET", "/catalog/categories", token=token)
        items = body.get("items", body) if isinstance(body, dict) else body
        if not isinstance(items, list):
            items = [items] if isinstance(items, dict) else []
        return [_map_category(c).model_dump() for c in items]
