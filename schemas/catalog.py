"""Catalog view DTOs — mirror the frontend's ProductGrid / ProductCard / ProductDetail props."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PriceView(BaseModel):
    """Canonical price shape. Frontend expects `{amount, currency}`, not backend's flat fields."""
    amount: float = Field(..., description="Price amount in the smallest currency unit or major unit")
    currency: str = Field(..., description="ISO 4217 currency code, e.g. 'NGN', 'USD'")


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
