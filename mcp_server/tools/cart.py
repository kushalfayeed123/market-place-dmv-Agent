"""Cart tools — session-scoped cart operations.

The cart lives in the Gateway session (Redis), NOT in the backend. These tools
mutate the session cart and return CartView DTOs.
"""

from __future__ import annotations

from typing import Any

from schemas.cart import CartLineItemView, CartView, PriceView


def register_cart_tools(server: Any, session: Any) -> None:
    """Register cart tools. `session` is the Gateway session object."""

    @server.tool(
        name="add_to_cart",
        description="Add a product variant to the session cart. Low risk, no money moved.",
    )
    async def add_to_cart(
        variant_id: str,
        product_id: str,
        product_name: str,
        variant_name: str,
        sku: str,
        quantity: int = 1,
        unit_price_amount: float = 0,
        unit_price_currency: str = "NGN",
        quantity_available: int = 0,
    ) -> dict:
        """Add an item to the session cart."""
        cart = session.get_cart()

        # Check if already in cart — update quantity
        for item in cart["items"]:
            if item["variant_id"] == variant_id:
                item["quantity"] += quantity
                item["line_total"]["amount"] = (
                    item["quantity"] * item["unit_price"]["amount"]
                )
                session.set_cart(cart)
                return _build_cart_view(cart).model_dump()

        # Add new line item
        line_total_amount = quantity * unit_price_amount
        cart["items"].append(
            {
                "variant_id": variant_id,
                "product_id": product_id,
                "product_name": product_name,
                "variant_name": variant_name,
                "sku": sku,
                "quantity": quantity,
                "unit_price": {
                    "amount": unit_price_amount,
                    "currency": unit_price_currency,
                },
                "line_total": {
                    "amount": line_total_amount,
                    "currency": unit_price_currency,
                },
                "quantity_available": quantity_available,
            }
        )
        session.set_cart(cart)
        return _build_cart_view(cart).model_dump()

    @server.tool(
        name="view_cart",
        description="View current session cart contents. Read-only.",
    )
    async def view_cart() -> dict:
        """Return the current session cart."""
        cart = session.get_cart()
        return _build_cart_view(cart).model_dump()

    @server.tool(
        name="remove_from_cart",
        description="Remove an item from the session cart.",
    )
    async def remove_from_cart(variant_id: str) -> dict:
        """Remove a line item from the cart by variant_id."""
        cart = session.get_cart()
        cart["items"] = [i for i in cart["items"] if i["variant_id"] != variant_id]
        session.set_cart(cart)
        return _build_cart_view(cart).model_dump()

    @server.tool(
        name="clear_cart",
        description="Clear all items from the session cart.",
    )
    async def clear_cart() -> dict:
        """Remove all items from the cart."""
        session.clear_cart()
        cart = session.get_cart()
        return _build_cart_view(cart).model_dump()


def _build_cart_view(cart: dict) -> CartView:
    """Build a CartView from the raw session cart dict."""
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
        for i in cart.get("items", [])
    ]

    # Compute subtotal
    total_amount = sum(i.line_total.amount for i in items)
    currency = items[0].line_total.currency if items else "NGN"

    # Build warnings
    warnings: list[str] = []
    for i in items:
        if i.quantity_available > 0 and i.quantity > i.quantity_available:
            warnings.append(
                f"{i.product_name} ({i.variant_name}): only {i.quantity_available} available"
            )

    return CartView(
        items=items,
        subtotal=PriceView(amount=total_amount, currency=currency),
        item_count=sum(i.quantity for i in items),
        currency=currency,
        is_empty=len(items) == 0,
        warnings=warnings,
    )
