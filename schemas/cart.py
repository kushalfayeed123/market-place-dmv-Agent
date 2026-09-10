"""Cart view DTOs — mirror the frontend's CartSummary props.

The cart lives in the Gateway session (Redis), NOT in the backend. These DTOs
represent the session-cart state that the frontend renders.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from .catalog import PriceView


class CartLineItemView(BaseModel):
    """A single line item in the cart."""
    variant_id: str
    product_id: str
    product_name: str
    variant_name: str
    sku: str
    quantity: int = Field(..., ge=1)
    unit_price: PriceView
    line_total: PriceView
    quantity_available: int = Field(..., ge=0, description="Available stock at time of add")


class CartView(BaseModel):
    """The full cart — rendered as CartSummary."""
    items: list[CartLineItemView] = Field(default_factory=list)
    subtotal: PriceView
    item_count: int = Field(..., ge=0, description="Total number of items (sum of quantities)")
    currency: str
    is_empty: bool = True
    warnings: list[str] = Field(default_factory=list, description="e.g. ['Item X is now out of stock']")
