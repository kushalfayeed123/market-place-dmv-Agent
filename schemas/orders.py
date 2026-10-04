"""Order view DTOs — mirror the frontend's OrderList / OrderDetail / OrderConfirmation props."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
from .catalog import PriceView


class OrderItemView(BaseModel):
    """A single line item within an order."""
    variant_id: str
    product_id: str
    product_name: str
    variant_name: str
    sku: str
    quantity: int
    unit_price: PriceView
    line_total: PriceView


class OrderSummaryView(BaseModel):
    """A single row in OrderList."""
    id: str
    order_number: str
    status: str
    total: PriceView
    item_count: int
    created_at: str
    merchant_id: str
    merchant_name: str


class OrderDetailView(OrderSummaryView):
    """Full order detail — rendered as OrderDetail / OrderConfirmation."""
    items: list[OrderItemView]
    subtotal: PriceView
    shipping_address: Optional[dict] = None
    payment_status: Optional[str] = None
    fulfillment_status: Optional[str] = None
    updated_at: Optional[str] = None


class CheckoutView(BaseModel):
    """Result of a successful initiate_checkout call."""
    order_id: str
    status: str
    total: PriceView
    item_count: int
    payment_required: bool = True
    message: str = "Checkout initiated. Awaiting payment."
