"""Canonical view DTOs — the single source of truth shared with the frontend.

Every MCP tool outputs one of these Pydantic models. The frontend mirrors each
as a Zod schema and validates `ui_directive.props` against it. When these change,
the frontend Zod schemas must change in lockstep (enforced by contract tests).
"""

from .catalog import ProductCardView, ProductDetailView, CategoryView, VariantView
from .cart import CartView, CartLineItemView
from .orders import OrderSummaryView, OrderDetailView, CheckoutView
from .payments import PaymentView, RefundView
from .ledger import BalanceView, LedgerEntryView
from .components import UIDirective, ComponentName

__all__ = [
    "ProductCardView",
    "ProductDetailView",
    "CategoryView",
    "VariantView",
    "CartView",
    "CartLineItemView",
    "OrderSummaryView",
    "OrderDetailView",
    "CheckoutView",
    "PaymentView",
    "RefundView",
    "BalanceView",
    "LedgerEntryView",
    "UIDirective",
    "ComponentName",
]
