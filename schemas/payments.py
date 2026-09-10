"""Payment view DTOs — mirror the frontend's PaymentCapturePanel / RefundStatus props."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
from .catalog import PriceView


class PaymentView(BaseModel):
    """A payment transaction — rendered in PaymentCapturePanel or payment status views."""
    id: str
    order_id: str
    status: str = Field(..., description="e.g. 'initialized', 'success', 'failed', 'pending'")
    amount: PriceView
    provider: Optional[str] = Field(None, description="e.g. 'paystack'")
    provider_reference: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RefundView(BaseModel):
    """A refund record."""
    id: str
    payment_id: str
    order_id: str
    status: str
    amount: PriceView
    reason: Optional[str] = None
    created_at: Optional[str] = None
