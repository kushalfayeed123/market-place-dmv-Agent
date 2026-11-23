"""Tool contract tests."""
import pytest
from schemas.catalog import ProductCardView, ProductDetailView, CategoryView, PriceView
from schemas.cart import CartView, CartLineItemView
from schemas.orders import OrderSummaryView, OrderDetailView, CheckoutView
from schemas.components import UIDirective, ComponentName

N = "₦"

class TestPriceView:
    def test_amount_is_int_not_float(self):
        p = PriceView(amount=100800, currency="NGN", display=N+"1,008")
        assert isinstance(p.amount, int)
        assert not isinstance(p.amount, float)

    def test_serializes_correctly(self):
        p = PriceView(amount=18500, currency="NGN", display=N+"185")
        d = p.model_dump()
        assert d["amount"] == 18500
        assert d["display"] == N+"185"

    def test_requires_display(self):
        with pytest.raises(Exception):
            PriceView(amount=100, currency="NGN")

class TestProductCardView:
    def test_minimal(self):
        c = ProductCardView(id="p1", name="Case", merchant_id="m1", merchant_name="S",
            price=PriceView(amount=5000, currency="NGN", display=N+"50"))
        d = c.model_dump()
        assert d["price"]["amount"] == 5000
        assert isinstance(d["price"]["amount"], int)

class TestCartView:
    def test_cart_item_amount_is_int(self):
        item = CartLineItemView(
            variant_id="v1", product_id="p1", product_name="X", variant_name="Y",
            sku="S1", quantity=1,
            unit_price=PriceView(amount=5000, currency="NGN", display=N+"50"),
            line_total=PriceView(amount=10000, currency="NGN", display=N+"100"),
            quantity_available=10,
        )
        d = item.model_dump()
        assert isinstance(d["unit_price"]["amount"], int)
        assert isinstance(d["line_total"]["amount"], int)

    def test_empty_cart(self):
        c = CartView(items=[], subtotal=PriceView(amount=0, currency="NGN", display=N+"0"),
            item_count=0, currency="NGN", is_empty=True)
        assert c.is_empty is True

class TestCheckoutView:
    def test_checkout(self):
        v = CheckoutView(order_id="o1", order_number="ORD-001", status="pending",
            total=PriceView(amount=18500, currency="NGN", display=N+"185"), item_count=3)
        assert v.order_id == "o1"
        assert v.payment_required is True


class TestOrderSummaryView:
    def test_merchant_name_defaults_to_empty_when_missing(self):
        """Regression: the buyer list_orders endpoint does not always populate
        merchant_name — it used to crash with a pydantic string_type validation
        error, so the buyer could not load their orders at all."""
        from mcp_server.tools.orders import _map_order_summary
        o = _map_order_summary({
            "id": "o1", "order_number": "ORD-001", "status": "paid",
            "total_amount": 100800, "currency": "NGN", "item_count": 2,
            "created_at": "2026-11-10T10:00:00Z", "merchant_id": None,
            "merchant_name": None,
        })
        assert o.merchant_name == ""
        assert o.merchant_id == ""
        assert o.total.amount == 100800

    def test_merchant_name_preserved_when_present(self):
        from mcp_server.tools.orders import _map_order_summary
        o = _map_order_summary({
            "id": "o1", "order_number": "ORD-001", "status": "paid",
            "total_amount": 100, "currency": "NGN", "merchant_name": "Leather Co",
        })
        assert o.merchant_name == "Leather Co"

class TestUIDirective:
    def test_minimal(self):
        d = UIDirective(component=ComponentName.ProductGrid, props={"items": []}, correlation_id="x")
        dumped = d.model_dump()
        assert dumped["component"] == "ProductGrid"

    def test_unknown_rejected(self):
        with pytest.raises(Exception):
            UIDirective(component="Bad", props={}, correlation_id="x")
