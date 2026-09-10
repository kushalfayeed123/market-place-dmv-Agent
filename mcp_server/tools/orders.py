"""Order tools — initiate_checkout, get_order_status, list_orders."""

from __future__ import annotations

from typing import Any

from mcp_server.backend_client import BackendClient, ToolError
from mcp_server.guards import ConfirmationGate, ConfirmationRequiredError
from mcp_server.idempotency import derive_idempotency_key
from schemas.cart import CartLineItemView, CartView
from schemas.orders import (
    CheckoutView,
    OrderDetailView,
    OrderItemView,
    OrderSummaryView,
    PriceView,
)


def _map_price(amount: float | None, currency: str | None) -> PriceView:
    return PriceView(amount=float(amount or 0), currency=currency or "NGN")


def _map_order_item(i: dict) -> OrderItemView:
    return OrderItemView(
        variant_id=str(i.get("variant_id", "")),
        product_id=str(i.get("product_id", "")),
        product_name=i.get("product_name", ""),
        variant_name=i.get("variant_name", ""),
        sku=i.get("sku", ""),
        quantity=i.get("quantity", 0),
        unit_price=_map_price(i.get("unit_price_amount"), i.get("unit_price_currency")),
        line_total=_map_price(i.get("line_total_amount"), i.get("line_total_currency")),
    )


def _map_order_summary(o: dict) -> OrderSummaryView:
    return OrderSummaryView(
        id=str(o.get("id", "")),
        status=o.get("status", ""),
        total=_map_price(o.get("total_amount"), o.get("total_currency")),
        item_count=o.get("item_count", len(o.get("items", []))),
        created_at=o.get("created_at", ""),
        merchant_id=str(o.get("merchant_id", "")),
        merchant_name=o.get("merchant_name", ""),
    )


def _map_order_detail(o: dict) -> OrderDetailView:
    items = [_map_order_item(i) for i in o.get("items", [])]
    return OrderDetailView(
        id=str(o.get("id", "")),
        status=o.get("status", ""),
        total=_map_price(o.get("total_amount"), o.get("total_currency")),
        item_count=o.get("item_count", len(items)),
        created_at=o.get("created_at", ""),
        merchant_id=str(o.get("merchant_id", "")),
        merchant_name=o.get("merchant_name", ""),
        items=items,
        subtotal=_map_price(o.get("subtotal_amount"), o.get("subtotal_currency")),
        payment_status=o.get("payment_status"),
        fulfillment_status=o.get("fulfillment_status"),
        updated_at=o.get("updated_at"),
    )


def register_orders_tools(
    server: Any,
    client: BackendClient,
    gate: ConfirmationGate,
    session: Any,
) -> None:
    """Register order tools."""

    @server.tool(
        name="initiate_checkout",
        description="Initiate checkout for the current cart. HIGH risk — financial, mutating. Requires user confirmation.",
    )
    async def initiate_checkout(
        user_confirmed_token: str | None = None,
        token: str | None = None,
    ) -> dict:
        """Initiate checkout — translates session cart to backend CheckoutRequest."""
        session_id = session.session_id
        turn_id = session.current_turn_id
        try:
            await gate.check_confirmation(session_id, turn_id, "initiate_checkout", user_confirmed_token)
        except ConfirmationRequiredError as e:
            raise ToolError(str(e), error_kind="confirmation_required") from e

        cart_raw = session.get_cart()
        items = [
            CartLineItemView(
                variant_id=i["variant_id"],
                product_id=i["product_id"],
                product_name=i["product_name"],
                variant_name=i["variant_name"],
                sku=i["sku"],
                quantity=i["quantity"],
                unit_price=PriceView(**i["unit_price"]),
                line_total=PriceView(**i["line_total"]),
                quantity_available=i.get("quantity_available", 0),
            )
            for i in cart_raw.get("items", [])
        ]
        cart = CartView(
            items=items,
            subtotal=PriceView(
                amount=sum(i.line_total.amount for i in items),
                currency=items[0].line_total.currency if items else "NGN",
            ),
            item_count=sum(i.quantity for i in items),
            currency=items[0].line_total.currency if items else "NGN",
            is_empty=len(items) == 0,
        )

        if cart.is_empty:
            raise ToolError("Cart is empty. Add items before checkout.", error_kind="validation_error")

        checkout_body = {
            "items": [
                {"variant_id": i.variant_id, "quantity": i.quantity}
                for i in cart.items
            ],
        }

        user_id = session.user_id
        idempotency_key = derive_idempotency_key(user_id, turn_id, "initiate_checkout", checkout_body)

        body = await client.call(
            "POST", "/orders/checkout",
            token=token,
            json_body=checkout_body,
            idempotency_key=idempotency_key,
        )

        session.clear_cart()

        return CheckoutView(
            order_id=str(body.get("id", body.get("order_id", ""))),
            status=body.get("status", "pending"),
            total=_map_price(body.get("total_amount"), body.get("total_currency")),
            item_count=body.get("item_count", cart.item_count),
            payment_required=body.get("payment_required", True),
            message=body.get("message", "Checkout initiated. Awaiting payment."),
        ).model_dump()

    @server.tool(
        name="get_order_status",
        description="Get the status of an order. Read-only.",
    )
    async def get_order_status(
        order_id: str,
        token: str | None = None,
    ) -> dict:
        """Get order status by ID."""
        body = await client.call(
            "GET", "/orders/{order_id}",
            token=token, path_params={"order_id": order_id},
        )
        return _map_order_detail(body).model_dump()

    @server.tool(
        name="list_orders",
        description="List orders for the current user (role-scoped). Read-only.",
    )
    async def list_orders(
        status: str | None = None,
        skip: int = 0,
        limit: int = 20,
        token: str | None = None,
    ) -> list[dict]:
        """List orders, optionally filtered by status."""
        params: dict[str, Any] = {"skip": skip, "limit": limit}
        if status:
            params["status"] = status
        body = await client.call("GET", "/orders/", token=token, params=params)
        items = body.get("items", body) if isinstance(body, dict) else body
        if not isinstance(items, list):
            items = [items] if isinstance(items, dict) else []
        return [_map_order_summary(o).model_dump() for o in items]