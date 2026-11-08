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
    inventory_policy: str = Field("tracked", description="'tracked' | 'untracked' | 'unlimited'")
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
    urls: Optional[list[str]] = None  # product image URLs (primary first, then gallery)
    quantity_available: Optional[int] = None
    inventory_policy: str = "tracked"  # 'tracked' | 'untracked' | 'unlimited'
    fulfillment_type: Optional[str] = None  # 'physical' | 'digital' | 'service'
    product_types: list[str] = Field(default_factory=list)
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
    qty, policy = _first_variant_stock(p)
    if policy is not None and policy != "tracked":
        in_stock = True  # untracked/unlimited variants never run out
    else:
        in_stock = (qty or 0) > 0 if qty is not None else None
    return {
        "id": str(p.get("id", "")),
        "name": p.get("title", p.get("name", "")),
        "price_display": format_money(price_minor, currency),
        "in_stock": in_stock,
        "status": p.get("status", "active"),
    }


def _first_variant_stock(p: dict) -> tuple[Optional[int], Optional[str]]:
    """Return (quantity_available, inventory_policy) from the first variant.

    quantity_available is only meaningful when the policy is 'tracked';
    untracked/unlimited variants report None so they are never "sold out".
    """
    qty = p.get("quantity_available")
    variants = p.get("variants")
    if isinstance(variants, list) and variants and isinstance(variants[0], dict):
        first = variants[0]
        policy = first.get("inventory_policy") or "tracked"
        if policy == "tracked":
            qty = first.get("quantity_available", qty)
        else:
            qty = None
        return qty, policy
    return qty, None


def product_ui(p: dict) -> dict:
    """Transform a backend product dict into the UI-directive prop shape.

    ``price.amount`` is an integer in minor units (kobo / cents); the
    frontend's ``formatMoney`` divides by 100 when rendering.
    """
    price_minor = p.get("base_price_amount", 0) or 0
    currency = p.get("base_price_currency", "NGN")
    # Derive quantity from first variant if the product is variant-based.
    # Only 'tracked' variants have a stock count; others report undefined so
    # the UI never shows them as sold out.
    qty, policy = _first_variant_stock(p)
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
        "inventory_policy": policy or "tracked",
        "fulfillment_type": p.get("fulfillment_type"),
        "product_types": p.get("product_types") or ([p.get("fulfillment_type")] if p.get("fulfillment_type") else []),
        "status": p.get("status", "active"),
    }
