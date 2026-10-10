"""Catalog view DTOs — mirror the frontend's ProductGrid / ProductCard / ProductDetail props."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PriceView(BaseModel):
    """Canonical price shape. Frontend expects `{amount, currency}`, not backend's flat fields.

    ``amount`` is always an **integer** in the smallest currency unit (minor units
    — kobo for NGN, cents for USD).  The frontend divides by 100 when rendering.
    ``display`` is the pre-formatted string shown to the user (e.g. "₦1,008")
    and should be used by the model when stating prices in its text output.
    """

    amount: int = Field(
        ..., description="Price amount in the smallest currency unit (minor units, e.g. kobo). Never a float."
    )
    currency: str = Field(
        ..., description="ISO 4217 currency code, e.g. 'NGN', 'USD'"
    )
    display: str = Field(
        ..., description="Pre-formatted display string, e.g. '₦1,008' or '₦1,008.50'"
    )


class VariantView(BaseModel):
    """A product variant as shown in VariantSelector."""
    id: str
    sku: str
    name: str
    price: PriceView
    quantity_available: Optional[int] = Field(None, description="None if stock not tracked")
    attributes: dict[str, str] = Field(default_factory=dict, description="e.g. {'color': 'red', 'size': 'L'}")


class ProductCardView(BaseModel):
    """A single card in ProductGrid. Minimal fields for listing."""
    id: str
    name: str
    merchant_id: str
    merchant_name: str
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    price: PriceView
    image_url: Optional[str] = None
    quantity_available: Optional[int] = None
    status: str = "active"


class ProductDetailView(ProductCardView):
    """Full product detail — extends the card with description, variants, attributes."""
    description: Optional[str] = None
    variants: list[VariantView] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CategoryView(BaseModel):
    """A product category."""
    id: str
    name: str
    slug: str
    parent_id: Optional[str] = None
    product_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Product serializers — single source of truth for transforming a raw backend
# ``ProductResponse`` dict into either an LLM-facing or UI-facing shape.
# ---------------------------------------------------------------------------


def product_llm(p: dict) -> dict:
    """Transform a backend product dict into the LLM-facing view.

    The model only ever sees display strings — **no raw amount integer**.
    This prevents the 100× price bug where the model echoes ``100800`` as
    "₦100,800" instead of "₦1,008".
    """
    from mcp_server.money import format_money

    price_minor = p.get("base_price_amount", 0) or 0
    currency = p.get("base_price_currency", "NGN")
    qty = p.get("quantity_available")
    return {
        "id": str(p.get("id", "")),
        "name": p.get("title", p.get("name", "")),
        "price_display": format_money(price_minor, currency),
        "in_stock": (qty or 0) > 0 if qty is not None else None,
        "status": p.get("status", "active"),
    }


def product_ui(p: dict) -> dict:
    """Transform a backend product dict into the UI-directive prop shape.

    ``price.amount`` is an integer in minor units (kobo / cents); the
    frontend's ``formatMoney`` divides by 100 when rendering.
    """
    price_minor = p.get("base_price_amount", 0) or 0
    currency = p.get("base_price_currency", "NGN")
    # Derive quantity from first variant if the product is variant-based.
    qty = p.get("quantity_available")
    if qty is None and isinstance(p.get("variants"), list) and p["variants"]:
        first_variant = p["variants"][0]
        if isinstance(first_variant, dict):
            qty = first_variant.get("quantity_available")
    return {
        "id": str(p.get("id", "")),
        "name": p.get("title", p.get("name", "")),
        "merchant_id": str(p.get("merchant_id", "")),
        "merchant_name": p.get("merchant_name") or p.get("store_name") or "",
        "category_id": str(p.get("category_id", "") or ""),
        "category_name": p.get("category_name"),
        "price": {
            "amount": int(price_minor),
            "currency": currency,
        },
        "image_url": p.get("image_url", p.get("urls", [None])[0] if isinstance(p.get("urls"), list) else None),
        "urls": p.get("urls"),
        "quantity_available": qty,
        "status": p.get("status", "active"),
    }
