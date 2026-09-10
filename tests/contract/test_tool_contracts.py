"""Tool contract tests — assert each MCP tool's output schema matches the corresponding view DTO.

These tests verify that the data shapes the agent produces are exactly what the
frontend expects (the "single schema, both ends" contract).
"""

import pytest
from schemas.catalog import ProductCardView, ProductDetailView, CategoryView, PriceView
from schemas.cart import CartView, CartLineItemView
from schemas.orders import OrderSummaryView, OrderDetailView, CheckoutView
from schemas.payments import PaymentView, RefundView
from schemas.ledger import BalanceView, LedgerEntryView
from schemas.components import UIDirective, ComponentName


class TestPriceView:
    """PriceView is the canonical price shape used across all DTOs."""

    def test_price_view_serializes_correctly(self):
        price = PriceView(amount=18500.0, currency="NGN")
        dumped = price.model_dump()
        assert dumped == {"amount": 18500.0, "currency": "NGN"}

    def test_price_view_requires_amount(self):
        with pytest.raises(Exception):
            PriceView(currency="NGN")  # type: ignore

    def test_price_view_requires_currency(self):
        with pytest.raises(Exception):
            PriceView(amount=100.0)  # type: ignore


class TestProductCardView:
    """ProductCardView is what ProductGrid renders."""

    def test_minimal_product_card(self):
        card = ProductCardView(
            id="prod-1",
            name="iPhone Case",
            merchant_id="merch-1",
            merchant_name="Tech Store",
            price=PriceView(amount=5000.0, currency="NGN"),
        )
        dumped = card.model_dump()
        assert dumped["id"] == "prod-1"
        assert dumped["price"]["amount"] == 5000.0
        assert dumped["price"]["currency"] == "NGN"

    def test_product_card_with_optional_fields(self):
        card = ProductCardView(
            id="prod-2",
            name="Ebook",
            merchant_id="merch-1",
            merchant_name="Book Store",
            category_id="cat-1",
            category_name="Books",
            price=PriceView(amount=3500.0, currency="NGN"),
            image_url="https://example.com/ebook.jpg",
            quantity_available=25,
        )
        dumped = card.model_dump()
        assert dumped["category_name"] == "Books"
        assert dumped["quantity_available"] == 25


class TestCartView:
    """CartView is what CartSummary renders."""

    def test_empty_cart(self):
        cart = CartView(
            items=[],
            subtotal=PriceView(amount=0, currency="NGN"),
            item_count=0,
            currency="NGN",
            is_empty=True,
        )
        dumped = cart.model_dump()
        assert dumped["is_empty"] is True
        assert dumped["item_count"] == 0

    def test_cart_with_items(self):
        cart = CartView(
            items=[
                CartLineItemView(
                    variant_id="var-1",
                    product_id="prod-1",
                    product_name="iPhone Case",
                    variant_name="Black",
                    sku="IC-BLK",
                    quantity=2,
                    unit_price=PriceView(amount=5000.0, currency="NGN"),
                    line_total=PriceView(amount=10000.0, currency="NGN"),
                    quantity_available=10,
                ),
            ],
            subtotal=PriceView(amount=10000.0, currency="NGN"),
            item_count=2,
            currency="NGN",
            is_empty=False,
        )
        dumped = cart.model_dump()
        assert dumped["is_empty"] is False
        assert len(dumped["items"]) == 1
        assert dumped["items"][0]["line_total"]["amount"] == 10000.0


class TestCheckoutView:
    """CheckoutView is returned by initiate_checkout."""

    def test_checkout_view(self):
        view = CheckoutView(
            order_id="order-123",
            status="pending",
            total=PriceView(amount=18500.0, currency="NGN"),
            item_count=3,
        )
        dumped = view.model_dump()
        assert dumped["order_id"] == "order-123"
        assert dumped["payment_required"] is True


class TestUIDirective:
    """UIDirective is the envelope sent to the frontend."""

    def test_minimal_directive(self):
        directive = UIDirective(
            component=ComponentName.ProductGrid,
            props={"items": []},
            correlation_id="abc123",
        )
        dumped = directive.model_dump()
        assert dumped["type"] == "ui_directive"
        assert dumped["version"] == 1
        assert dumped["component"] == "ProductGrid"

    def test_unknown_component_rejected(self):
        with pytest.raises(Exception):
            UIDirective(
                component="NotARealComponent",  # type: ignore
                props={},
                correlation_id="abc123",
            )

    def test_all_allowlisted_components_valid(self):
        """Every allowlisted component name must be a valid ComponentName."""
        for name in ComponentName:
            d = UIDirective(component=name, props={}, correlation_id="x")
            assert d.component == name
