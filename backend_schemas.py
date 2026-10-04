"""Thin re-export of backend Pydantic schemas.

This module imports the backend's pure Pydantic schema classes so the MCP tools'
input/output types are literally the same classes the backend validates against.
This keeps the tool contract and the backend contract in lockpoint.

Usage: run with PYTHONPATH including the backend directory, e.g.
    PYTHONPATH=../backend python -m gateway.main
or install the backend as an editable dependency.
"""

from __future__ import annotations

# Auth
from app.schemas.auth import (  # type: ignore[import-untyped]
    UserResponse,
    UserUpdate,
    TokenResponse,
    UserLogin as LoginRequest,
    UserRegister as RegisterRequest,
    PasswordChange,
    PasswordResetConfirm,
)

# Catalog
from app.schemas.catalog import (  # type: ignore[import-untyped]
    ProductResponse,
    ProductCreate,
    ProductVariantResponse,
    ProductVariantCreate,
    CategoryResponse,
    InventoryUpdate,
)

# Orders
from app.schemas.orders import (  # type: ignore[import-untyped]
    CheckoutRequest,
    CheckoutResponse,
    OrderResponse,
)

# Payments
from app.schemas.payments import (  # type: ignore[import-untyped]
    PaymentResponse as PaymentTransactionResponse,
    PaymentResponse,
    RefundRequest,
    RefundResponse,
)

# Ledger
from app.schemas.ledger import (  # type: ignore[import-untyped]
    LedgerEntryResponse,
    LedgerBalanceResponse,
)

# Merchants
from app.schemas.merchants import (  # type: ignore[import-untyped]
    MerchantResponse,
    MerchantCreate,
    MerchantUpdate,
    MerchantPayoutAccountResponse,
    MerchantPayoutAccountCreate,
    MerchantPayoutAccountUpdate,
)

# Fulfillment
from app.schemas.fulfillment import (  # type: ignore[import-untyped]
    FulfillmentResponse,
    FulfillmentCreate,
    FulfillmentStatusUpdate,
)

# Store
from app.schemas.store import (  # type: ignore[import-untyped]
    StoreResponse,
    StoreCreate,
)

__all__ = [
    # Auth
    "UserResponse",
    "UserUpdate",
    "TokenResponse",
    "LoginRequest",
    "RegisterRequest",
    "PasswordChange",
    "PasswordResetConfirm",
    # Catalog
    "ProductResponse",
    "ProductCreate",
    "ProductVariantResponse",
    "ProductVariantCreate",
    "CategoryResponse",
    "InventoryUpdate",
    # Orders
    "CheckoutRequest",
    "CheckoutResponse",
    "OrderResponse",
    # Payments
    "PaymentTransactionResponse",
    "RefundRequest",
    "RefundResponse",
    # Ledger
    "LedgerEntryResponse",
    "LedgerBalanceResponse",
    # Merchants
    "MerchantResponse",
    "MerchantCreate",
    "MerchantUpdate",
    "MerchantPayoutAccountResponse",
    "MerchantPayoutAccountCreate",
    "MerchantPayoutAccountUpdate",
    # Fulfillment
    "FulfillmentResponse",
    "FulfillmentCreate",
    "FulfillmentStatusUpdate",
    # Store
    "StoreResponse",
    "StoreCreate",
]
