"""UI directive envelope + allowlisted component names.

The frontend renders `ui_directive` messages from the agent. The `component`
field MUST be one of the allowlisted names; anything else falls back to a
generic error component. Props are validated against the corresponding view DTO.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ComponentName(str, Enum):
    """Allowlisted component names. The frontend has a registry mapping these to React components."""
    # Catalog
    ProductGrid = "ProductGrid"
    ProductCard = "ProductCard"
    ProductDetail = "ProductDetail"
    CategoryList = "CategoryList"
    VariantSelector = "VariantSelector"
    # Cart & Checkout
    CartSummary = "CartSummary"
    ConfirmationDialog = "ConfirmationDialog"
    PaymentCapturePanel = "PaymentCapturePanel"
    OrderConfirmation = "OrderConfirmation"
    # Orders
    OrderList = "OrderList"
    OrderDetail = "OrderDetail"
    # Merchant / Ledger
    MerchantBalanceCard = "MerchantBalanceCard"
    LedgerTable = "LedgerTable"
    MerchantProfile = "MerchantProfile"
    # Fulfillment
    FulfillmentTracker = "FulfillmentTracker"
    # Auth / system
    SignInPrompt = "SignInPrompt"
    ErrorMessage = "ErrorMessage"


class UIDirective(BaseModel):
    """The envelope the agent sends to the frontend to render a component."""
    type: str = "ui_directive"
    version: int = 1
    component: ComponentName
    props: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = Field(..., description="Correlation id tracing this directive back to a tool call")
