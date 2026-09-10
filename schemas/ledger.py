"""Ledger view DTOs — mirror the frontend's MerchantBalanceCard / LedgerTable props."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
from .catalog import PriceView


class BalanceView(BaseModel):
    """Merchant balance — rendered as MerchantBalanceCard."""
    merchant_id: str
    merchant_name: str
    available: PriceView
    pending: PriceView
    currency: str
    updated_at: Optional[str] = None


class LedgerEntryView(BaseModel):
    """A single ledger entry row — rendered in LedgerTable."""
    id: str
    entry_type: str = Field(..., description="e.g. 'sale', 'refund', 'payout_hold', 'payout_paid', 'commission'")
    amount: PriceView
    balance_after: Optional[PriceView] = None
    order_id: Optional[str] = None
    payment_id: Optional[str] = None
    description: Optional[str] = None
    created_at: str
    entry_group_id: Optional[str] = None
