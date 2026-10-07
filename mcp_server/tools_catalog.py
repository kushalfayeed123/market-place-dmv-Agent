"""Tool catalog — metadata for every MCP tool.

Maps each tool name to its backend endpoint, risk tier, whether confirmation
is required, and which roles may see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RiskTier(str, Enum):
    READ_ONLY = "read_only"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


ROLES = frozenset({"buyer", "merchant_owner", "merchant_staff", "platform_admin"})


@dataclass(frozen=True)
class ToolMeta:
    """Static metadata for one tool."""
    name: str
    method: str
    path: str
    risk_tier: RiskTier
    requires_confirmation: bool
    visible_to: frozenset[str]
    description: str


# ── Read-only tools ─────────────────────────────────────────────────────
SEARCH_PRODUCTS = ToolMeta(
    name="search_products",
    method="GET",
    path="/catalog/products",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=ROLES | frozenset({"anonymous"}),
    description="Search the product catalog. Read-only, safe, no confirmation needed.",
)

GET_PRODUCT_DETAIL = ToolMeta(
    name="get_product_detail",
    method="GET",
    path="/catalog/products/{product_id}",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=ROLES | frozenset({"anonymous"}),
    description="Get full product detail including variants. Read-only.",
)

GET_CATEGORIES = ToolMeta(
    name="get_categories",
    method="GET",
    path="/catalog/categories",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=ROLES | frozenset({"anonymous"}),
    description="List product categories. Read-only.",
)

# ── RAG / knowledge tools ─────────────────────────────────────────────
SEMANTIC_SEARCH = ToolMeta(
    name="semantic_search",
    method="VECTOR",
    path="",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=ROLES | frozenset({"anonymous"}),
    description="Semantic search over marketplace knowledge (products, policies, FAQs). Read-only.",
)

GET_USER_PROFILE = ToolMeta(
    name="get_user_profile",
    method="GET",
    path="/auth/me",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=ROLES,
    description="Get the current authenticated user's profile. Read-only.",
)
# ── Cart tools (session-scoped) ─────────────────────────────────────────
ADD_TO_CART = ToolMeta(
    name="add_to_cart",
    method="SESSION",
    path="",
    risk_tier=RiskTier.LOW,
    requires_confirmation=False,
    visible_to=ROLES | frozenset({"anonymous"}),
    description="Add a product variant to the session cart. Low risk, no money moved.",
)

VIEW_CART = ToolMeta(
    name="view_cart",
    method="SESSION",
    path="",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=ROLES | frozenset({"anonymous"}),
    description="View current session cart contents. Read-only.",
)

REMOVE_FROM_CART = ToolMeta(
    name="remove_from_cart",
    method="SESSION",
    path="",
    risk_tier=RiskTier.LOW,
    requires_confirmation=False,
    visible_to=ROLES | frozenset({"anonymous"}),
    description="Remove an item from the session cart.",
)

CLEAR_CART = ToolMeta(
    name="clear_cart",
    method="SESSION",
    path="",
    risk_tier=RiskTier.LOW,
    requires_confirmation=False,
    visible_to=ROLES | frozenset({"anonymous"}),
    description="Clear all items from the session cart.",
)

# ── Order tools ─────────────────────────────────────────────────────────
INITIATE_CHECKOUT = ToolMeta(
    name="initiate_checkout",
    method="POST",
    path="/orders/checkout",
    risk_tier=RiskTier.HIGH,
    requires_confirmation=True,
    visible_to=frozenset({"buyer", "platform_admin"}),
    description="Initiate checkout. HIGH risk — financial, mutating. Requires user confirmation.",
)

GET_ORDER_STATUS = ToolMeta(
    name="get_order_status",
    method="GET",
    path="/orders/{order_id}",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=ROLES,
    description="Get the status of an order. Read-only.",
)

# ── Payment tools ───────────────────────────────────────────────────────
PROCESS_PAYMENT = ToolMeta(
    name="process_payment",
    method="POST",
    path="/payments/process",
    risk_tier=RiskTier.HIGH,
    requires_confirmation=True,
    visible_to=frozenset({"buyer", "platform_admin"}),
    description="Process payment for an order. HIGH risk — financial. Requires user confirmation.",
)

GET_PAYMENT_STATUS = ToolMeta(
    name="get_payment_status",
    method="GET",
    path="/payments/{payment_id}",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=ROLES,
    description="Get payment status. Read-only.",
)

LIST_PAYMENTS = ToolMeta(
    name="list_payments",
    method="GET",
    path="/payments/",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=ROLES,
    description="List payments with optional filtering. Read-only.",
)

REQUEST_REFUND = ToolMeta(
    name="request_refund",
    method="POST",
    path="/payments/{payment_id}/refund",
    risk_tier=RiskTier.HIGH,
    requires_confirmation=True,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"}),
    description="Request a refund. HIGH risk — financial. Requires user confirmation.",
)

# ── Ledger tools ────────────────────────────────────────────────────────
GET_MERCHANT_BALANCE = ToolMeta(
    name="get_merchant_balance",
    method="GET",
    path="/ledger/balance/{merchant_id}",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"}),
    description="Get merchant balance. Read-only, merchant-scoped.",
)

GET_MERCHANT_LEDGER = ToolMeta(
    name="get_merchant_ledger",
    method="GET",
    path="/ledger/entries",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"}),
    description="Get merchant ledger entries. Read-only, merchant-scoped.",
)

# ── Fulfillment tools ───────────────────────────────────────────────────
GET_FULFILLMENT_STATUS = ToolMeta(
    name="get_fulfillment_status",
    method="GET",
    path="/fulfillment/order/{order_id}",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=ROLES,
    description="Get fulfillment status for an order. Read-only.",
)

LIST_FULFILLMENTS = ToolMeta(
    name="list_fulfillments",
    method="GET",
    path="/fulfillment/",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"}),
    description="List fulfillments for a merchant. Read-only.",
)

CREATE_FULFILLMENT = ToolMeta(
    name="create_fulfillment",
    method="POST",
    path="/fulfillment/",
    risk_tier=RiskTier.MEDIUM,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"}),
    description="Create a fulfillment record. Medium risk — mutating, non-financial.",
)

UPDATE_FULFILLMENT_STATUS = ToolMeta(
    name="update_fulfillment_status",
    method="PUT",
    path="/fulfillment/{fulfillment_id}/status",
    risk_tier=RiskTier.MEDIUM,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"}),
    description="Update fulfillment status. Medium risk — mutating, non-financial.",
)

# ── Merchant tools ──────────────────────────────────────────────────────
LIST_ORDERS = ToolMeta(
    name="list_orders",
    method="GET",
    path="/orders/",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=ROLES,
    description="List orders for the current user (role-scoped). Read-only.",
)


# ── Merchant tools ────────────────────────────────────────────────────
GET_MERCHANT_ORDERS = ToolMeta(
    name="get_merchant_orders",
    method="GET",
    path="/merchants/{merchant_id}/orders",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"}),
    description="List orders for a merchant. Read-only, merchant-scoped.",
)

GET_MERCHANT_PROFILE = ToolMeta(
    name="get_merchant_profile",
    method="GET",
    path="/merchants/{merchant_id}",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "platform_admin"}),
    description="Get merchant profile. Read-only.",
)

LIST_MERCHANTS = ToolMeta(
    name="list_merchants",
    method="GET",
    path="/merchants/",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=frozenset({"platform_admin"}),
    description="List all merchants. Admin only. Read-only.",
)

# ── Product management (merchant) ─────────────────────────────────────
CREATE_PRODUCT = ToolMeta(
    name="create_product",
    method="POST",
    path="/catalog/products",
    risk_tier=RiskTier.MEDIUM,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"}),
    description="Create a new product. Medium risk — mutating, non-financial.",
)

UPDATE_INVENTORY = ToolMeta(
    name="update_inventory",
    method="PUT",
    path="/catalog/inventory/{variant_id}",
    risk_tier=RiskTier.MEDIUM,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"}),
    description="Update inventory for a variant. Medium risk — mutating, non-financial.",
)

# ── Store tools ──
GET_STORE_BY_MERCHANT = ToolMeta(
    name="get_store_by_merchant",
    method="GET",
    path="/stores/merchant/{merchant_id}",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"} ),
    description="Get the store for a merchant. Read-only, merchant-scoped.",
)

# ── Payout account tools ──
GET_PAYOUT_ACCOUNTS = ToolMeta(
    name="get_payout_accounts",
    method="GET",
    path="/merchants/{merchant_id}/payout-accounts",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"} ),
    description="List payout accounts for a merchant. Read-only, merchant-scoped.",
)

CREATE_PAYOUT_ACCOUNT = ToolMeta(
    name="create_payout_account",
    method="POST",
    path="/merchants/{merchant_id}/payout-accounts",
    risk_tier=RiskTier.MEDIUM,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"} ),
    description="Create a payout account for a merchant. Medium risk - mutating, non-financial.",
)


# ── Payout account CRUD tools ──
GET_PAYOUT_ACCOUNT = ToolMeta(
    name="get_payout_account",
    method="GET",
    path="/merchants/{merchant_id}/payout-accounts/{payout_id}",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"} ),
    description="Get a single payout account by id. Read-only, merchant-scoped.",
)

UPDATE_PAYOUT_ACCOUNT = ToolMeta(
    name="update_payout_account",
    method="PUT",
    path="/merchants/{merchant_id}/payout-accounts/{payout_id}",
    risk_tier=RiskTier.MEDIUM,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"} ),
    description="Update a payout account for a merchant. Medium risk - mutating, non-financial.",
)

DELETE_PAYOUT_ACCOUNT = ToolMeta(
    name="delete_payout_account",
    method="DELETE",
    path="/merchants/{merchant_id}/payout-accounts/{payout_id}",
    risk_tier=RiskTier.MEDIUM,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"} ),
    description="Delete (soft-delete) a payout account for a merchant. Medium risk - mutating, non-financial.",
)


# ── KYC status tools ──
GET_MERCHANT_KYC_STATUS = ToolMeta(
    name="get_merchant_kyc_status",
    method="GET",
    path="/merchants/{merchant_id}/kyc",
    risk_tier=RiskTier.READ_ONLY,
    requires_confirmation=False,
    visible_to=frozenset({"merchant_owner", "merchant_staff", "platform_admin"}),
    description="Get KYC status for a merchant. Read-only, merchant-scoped.",
)


# ── Admin tools ─────────────────────────────────────────────────────────
REVIEW_KYC = ToolMeta(
    name="review_kyc",
    method="PATCH",
    path="/admin/merchants/{merchant_id}/kyc",
    risk_tier=RiskTier.HIGH,
    requires_confirmation=True,
    visible_to=frozenset({"platform_admin"}),
    description="Review/approve/reject merchant KYC. HIGH risk — admin-only. Requires confirmation.",
)

REQUEST_PAYOUT = ToolMeta(
    name="request_payout",
    method="POST",
    path="/merchants/{merchant_id}/payouts",
    risk_tier=RiskTier.HIGH,
    requires_confirmation=True,
    visible_to=frozenset({"merchant_owner", "platform_admin"}),
    description="Request a payout. HIGH risk — financial. Requires confirmation + MFA step-up.",
)



# ── Registry ─────────────────────────────────────────────────────────────
ALL_TOOLS: tuple[ToolMeta, ...] = (
    SEARCH_PRODUCTS,
    GET_PRODUCT_DETAIL,
    GET_CATEGORIES,
    SEMANTIC_SEARCH,
    GET_USER_PROFILE,
    ADD_TO_CART,
    VIEW_CART,
    REMOVE_FROM_CART,
    CLEAR_CART,
    INITIATE_CHECKOUT,
    GET_ORDER_STATUS,
    LIST_ORDERS,
    PROCESS_PAYMENT,
    GET_PAYMENT_STATUS,
    LIST_PAYMENTS,
    REQUEST_REFUND,
    GET_MERCHANT_BALANCE,
    GET_MERCHANT_LEDGER,
    GET_FULFILLMENT_STATUS,
    LIST_FULFILLMENTS,
    CREATE_FULFILLMENT,
    UPDATE_FULFILLMENT_STATUS,
    GET_MERCHANT_ORDERS,
    GET_MERCHANT_PROFILE,
    LIST_MERCHANTS,
    GET_STORE_BY_MERCHANT,
    GET_PAYOUT_ACCOUNTS,
    CREATE_PAYOUT_ACCOUNT,
    GET_PAYOUT_ACCOUNT,
    UPDATE_PAYOUT_ACCOUNT,
    DELETE_PAYOUT_ACCOUNT,
    GET_MERCHANT_KYC_STATUS,
    CREATE_PRODUCT,
    UPDATE_INVENTORY,
    REVIEW_KYC,
    REQUEST_PAYOUT,
)

BY_NAME: dict[str, ToolMeta] = {t.name: t for t in ALL_TOOLS}


def tools_for_role(role: str | None) -> list[ToolMeta]:
    """Return the tools visible to the given role (cosmetic filter, not security)."""
    if role is None:
        role = "anonymous"
    return [t for t in ALL_TOOLS if role in t.visible_to]


def is_mutating_financial(tool_name: str) -> bool:
    """Whether the tool requires the confirmation gate."""
    meta = BY_NAME.get(tool_name)
    return meta.requires_confirmation if meta else False


# ── Parameter schemas (mirror the MCP server's tool signatures) ─────────────
# The gateway agent exposes these as the model's function-calling schemas so the
# model can fill arguments accurately instead of guessing. Sensitive/reserved
# keys (token, user_confirmed_token) are intentionally absent — they are injected
# server-side by the agent and never sent to the model.
TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "search_products": {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "Search keywords matched against product name, title, description, SKU, category, and merchant store name."},
            "category_id": {"type": "string", "description": "Restrict results to a single category id."},
            "merchant_id": {"type": "string", "description": "Restrict results to a single merchant/store."},
            "price_min": {"type": "number", "description": "Lower price bound (inclusive), in the requested currency."},
            "price_max": {"type": "number", "description": "Upper price bound (inclusive), in the requested currency."},
            "currency": {"type": "string", "description": "ISO currency code, e.g. 'NGN' or 'USD'.", "default": "NGN"},
            "skip": {"type": "integer", "description": "Pagination offset.", "default": 0},
            "limit": {"type": "integer", "description": "Page size.", "default": 20},
        },
        "required": ["q"],
    },
    "get_product_detail": {
        "type": "object",
        "properties": {"product_id": {"type": "string", "description": "The product id to look up."}},
        "required": ["product_id"],
    },
    "get_categories": {"type": "object", "properties": {}, "required": []},
    "semantic_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language description of what the user is looking for (e.g. 'red leather wallet', 'laptop charger under $50')."},
            "top_k": {"type": "integer", "description": "Maximum number of matches to return.", "default": 5},
        },
        "required": ["query"],
    },
    "get_user_profile": {"type": "object", "properties": {}, "required": []},
    "view_cart": {"type": "object", "properties": {}, "required": []},
    "add_to_cart": {
        "type": "object",
        "properties": {
            "variant_id": {"type": "string", "description": "The product variant to add."},
            "product_id": {"type": "string", "description": "The product id."},
            "product_name": {"type": "string", "description": "Display name of the product."},
            "variant_name": {"type": "string", "description": "Display name of the variant."},
            "sku": {"type": "string", "description": "Stock-keeping unit of the variant."},
            "quantity": {"type": "integer", "description": "Units to add.", "default": 1},
            "unit_price_amount": {"type": "number", "description": "Unit price amount in minor units are not needed; pass the major-unit amount."},
            "unit_price_currency": {"type": "string", "description": "ISO currency code.", "default": "NGN"},
            "quantity_available": {"type": "integer", "description": "Units available for the variant.", "default": 0},
        },
        "required": ["variant_id"],
    },
    "remove_from_cart": {
        "type": "object",
        "properties": {"variant_id": {"type": "string", "description": "The variant id to remove from the cart."}},
        "required": ["variant_id"],
    },
    "clear_cart": {"type": "object", "properties": {}, "required": []},
    "initiate_checkout": {
        "type": "object",
        "properties": {
            "items": {"type": "array", "description": "Cart items to checkout.", "items": {"type": "object"}},
            "shipping_address": {"type": "object", "description": "Shipping address for physical goods."},
        },
        "required": ["items"],
    },
    "get_order_status": {
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "The order id to look up."}},
        "required": ["order_id"],
    },
    "list_orders": {"type": "object", "properties": {}, "required": []},
    "process_payment": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order to pay for."},
            "provider": {"type": "string", "description": "Payment provider, e.g. 'paystack'.", "default": "paystack"},
        },
        "required": ["order_id"],
    },
    "get_payment_status": {
        "type": "object",
        "properties": {"payment_id": {"type": "string", "description": "The payment id."}},
        "required": ["payment_id"],
    },
    "list_payments": {"type": "object", "properties": {}, "required": []},
    "request_refund": {
        "type": "object",
        "properties": {
            "payment_id": {"type": "string", "description": "The payment to refund."},
            "amount": {"type": "integer", "description": "Refund amount in minor units."},
        },
        "required": ["payment_id"],
    },
    "get_merchant_balance": {
        "type": "object",
        "properties": {"merchant_id": {"type": "string", "description": "The merchant id."}},
        "required": ["merchant_id"],
    },
    "get_merchant_ledger": {
        "type": "object",
        "properties": {
            "merchant_id": {"type": "string", "description": "The merchant id."},
            "skip": {"type": "integer", "description": "Pagination offset.", "default": 0},
            "limit": {"type": "integer", "description": "Page size.", "default": 100},
        },
        "required": ["merchant_id"],
    },
    "get_fulfillment_status": {
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "The order id."}},
        "required": ["order_id"],
    },
    "list_fulfillments": {"type": "object", "properties": {}, "required": []},
    "create_fulfillment": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order to fulfill."},
            "tracking_code": {"type": "string", "description": "Tracking code for the shipment."},
            "courier": {"type": "string", "description": "Courier name."},
        },
        "required": ["order_id"],
    },
    "update_fulfillment_status": {
        "type": "object",
        "properties": {
            "fulfillment_id": {"type": "string", "description": "The fulfillment id."},
            "status": {"type": "string", "description": "New fulfillment status."},
        },
        "required": ["fulfillment_id", "status"],
    },
    "get_merchant_orders": {
        "type": "object",
        "properties": {"merchant_id": {"type": "string", "description": "The merchant id."}},
        "required": ["merchant_id"],
    },
    "get_merchant_profile": {
        "type": "object",
        "properties": {"merchant_id": {"type": "string", "description": "The merchant id to look up."}},
        "required": ["merchant_id"],
    },
    "list_merchants": {"type": "object", "properties": {}, "required": []},
    "get_store_by_merchant": {
        "type": "object",
        "properties": {"merchant_id": {"type": "string", "description": "The merchant id."}},
        "required": ["merchant_id"],
    },
    "get_payout_accounts": {
        "type": "object",
        "properties": {"merchant_id": {"type": "string", "description": "The merchant id."}},
        "required": ["merchant_id"],
    },
    "create_payout_account": {
        "type": "object",
        "properties": {
            "merchant_id": {"type": "string", "description": "The merchant id."},
            "provider": {"type": "string", "description": "Payout provider name."},
            "currency": {"type": "string", "description": "ISO currency code."},
            "external_ref": {"type": "string", "description": "Provider-side reference."},
            "account_last4": {"type": "string", "description": "Last 4 digits of bank account."},
            "bank_name": {"type": "string", "description": "Bank name."},
            "account_holder_name": {"type": "string", "description": "Account holder name."},
            "bank_code": {"type": "string", "description": "Bank code."},
            "routing_number": {"type": "string", "description": "Routing number."},
            "account_type": {"type": "string", "description": "Account type."},
            "country": {"type": "string", "description": "ISO 3166-1 alpha-2 country code."},
            "is_active": {"type": "boolean", "description": "Whether the account is active."},
        },
        "required": ["merchant_id", "provider", "currency", "external_ref"],
    },
    "create_product": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Product title."},
            "slug": {"type": "string", "description": "URL-safe slug."},
            "description": {"type": "string", "description": "Product description."},
            "fulfillment_type": {"type": "string", "description": "physical, digital, or service."},
            "base_price_amount": {"type": "integer", "description": "Price in minor units (e.g. kobo)."},
            "base_price_currency": {"type": "string", "description": "ISO currency code.", "default": "NGN"},
            "category_id": {"type": "string", "description": "Category id."},
            "merchant_id": {"type": "string", "description": "Merchant id (looked up if omitted)."},
            "store_id": {"type": "string", "description": "Store id (looked up if omitted)."},
            "urls": {"type": "array", "description": "Product image URLs.", "items": {"type": "string"}},
            "status": {"type": "string", "description": "draft, active, or suspended."},
        },
        "required": ["title"],
    },
    "update_inventory": {
        "type": "object",
        "properties": {
            "variant_id": {"type": "string", "description": "The product variant id."},
            "quantity_available": {"type": "integer", "description": "New available quantity."},
        },
        "required": ["variant_id", "quantity_available"],
    },
    "review_kyc": {
        "type": "object",
        "properties": {
            "merchant_id": {"type": "string", "description": "The merchant id."},
            "kyc_status": {"type": "string", "description": "New KYC status: pending, test_mode, verified, or rejected."},
            "reason": {"type": "string", "description": "Reason for the review."},
        },
        "required": ["merchant_id", "kyc_status"],
    },
    "request_payout": {
        "type": "object",
        "properties": {
            "merchant_id": {"type": "string", "description": "The merchant id."},
            "amount": {"type": "integer", "description": "Payout amount in minor units."},
            "currency": {"type": "string", "description": "ISO currency code.", "default": "NGN"},
        },
        "required": ["merchant_id", "amount"],
    },
    "get_payout_account": {
        "type": "object",
        "properties": {
            "merchant_id": {"type": "string", "description": "The merchant id."},
            "payout_id": {"type": "string", "description": "The payout account id."},
        },
        "required": ["merchant_id", "payout_id"],
    },
    "update_payout_account": {
        "type": "object",
        "properties": {
            "merchant_id": {"type": "string", "description": "The merchant id."},
            "payout_id": {"type": "string", "description": "The payout account id."},
            "provider": {"type": "string", "description": "Payout provider name."},
            "currency": {"type": "string", "description": "ISO currency code."},
            "external_ref": {"type": "string", "description": "Provider-side reference."},
            "account_last4": {"type": "string", "description": "Last 4 digits of bank account."},
            "bank_name": {"type": "string", "description": "Bank name."},
            "account_holder_name": {"type": "string", "description": "Account holder name."},
            "bank_code": {"type": "string", "description": "Bank code."},
            "routing_number": {"type": "string", "description": "Routing number."},
            "account_type": {"type": "string", "description": "Account type."},
            "country": {"type": "string", "description": "ISO 3166-1 alpha-2 country code."},
            "is_active": {"type": "boolean", "description": "Whether active."},
        },
        "required": ["merchant_id", "payout_id"],
    },
    "delete_payout_account": {
        "type": "object",
        "properties": {
            "merchant_id": {"type": "string", "description": "The merchant id."},
            "payout_id": {"type": "string", "description": "The payout account id."},
        },
        "required": ["merchant_id", "payout_id"],
    },
    "get_merchant_kyc_status": {
        "type": "object",
        "properties": {"merchant_id": {"type": "string", "description": "The merchant id."}},
        "required": ["merchant_id"],
    },
}




def parameters_for(tool_name: str) -> dict[str, Any]:
    """Return the OpenAI parameter schema for a tool (empty object schema if unknown)."""
    return TOOL_PARAMETERS.get(tool_name, {"type": "object", "properties": {}, "required": []})

