"""Tests for product_llm and product_ui serializers in backend/app/schemas/catalog.py.

These verify the money-correctness fix: the LLM-facing view must NOT contain
raw amount integers (which would trigger the 100x bug).
"""

import pytest
from schemas.catalog import product_llm, product_ui


class TestProductLLM:
    """product_llm must expose only display strings — no raw amount."""

    def test_no_raw_amount_in_output(self):
        p = {
            "id": "prod-1",
            "title": "iPhone Case",
            "merchant_name": "Tech Store",
            "base_price_amount": 100800,
            "base_price_currency": "NGN",
            "quantity_available": 5,
            "status": "active",
        }
        result = product_llm(p)
        dumped = repr(result)
        # The raw minor-unit integer must NOT appear anywhere in the LLM view
        assert "100800" not in dumped

    def test_has_price_display_string(self):
        p = {
            "id": "prod-1",
            "title": "iPhone Case",
            "merchant_name": "Tech Store",
            "base_price_amount": 100800,
            "base_price_currency": "NGN",
            "quantity_available": 5,
            "status": "active",
        }
        result = product_llm(p)
        assert result["price_display"] == "₦1,008"

    def test_has_in_stock_flag(self):
        p = {
            "id": "prod-1",
            "title": "Widget",
            "merchant_name": "Store",
            "base_price_amount": 5000,
            "base_price_currency": "NGN",
            "quantity_available": 3,
            "status": "active",
        }
        result = product_llm(p)
        assert result["in_stock"] is True

    def test_zero_stock_in_stock_false(self):
        p = {
            "id": "prod-1",
            "title": "Widget",
            "merchant_name": "Store",
            "base_price_amount": 5000,
            "base_price_currency": "NGN",
            "quantity_available": 0,
            "status": "active",
        }
        result = product_llm(p)
        assert result["in_stock"] is False

    def test_no_amount_key_at_all(self):
        p = {
            "id": "prod-1",
            "title": "Widget",
            "base_price_amount": 5000,
            "base_price_currency": "NGN",
            "quantity_available": 5,
            "status": "active",
        }
        result = product_llm(p)
        assert "amount" not in result
        assert "price" not in result  # no nested amount either


class TestProductUI:
    """product_ui must carry integer minor units for the frontend."""

    def test_amount_is_int(self):
        p = {
            "id": "prod-1",
            "title": "Widget",
            "merchant_id": "merch-1",
            "store_name": "My Store",
            "base_price_amount": 100800,
            "base_price_currency": "NGN",
            "status": "active",
        }
        result = product_ui(p)
        assert result["price"]["amount"] == 100800
        assert isinstance(result["price"]["amount"], int)

    def test_handles_null_category_name(self):
        p = {
            "id": "prod-1",
            "title": "Widget",
            "merchant_id": "merch-1",
            "base_price_amount": 5000,
            "base_price_currency": "NGN",
            "category_name": None,
            "status": "active",
        }
        result = product_ui(p)
        assert result["category_name"] is None

    def test_derives_qty_from_first_variant(self):
        p = {
            "id": "prod-1",
            "title": "Widget",
            "merchant_id": "merch-1",
            "base_price_amount": 5000,
            "base_price_currency": "NGN",
            "variants": [{"quantity_available": 7}],
            "status": "active",
        }
        result = product_ui(p)
        assert result["quantity_available"] == 7

    def test_image_url_from_urls_list(self):
        p = {
            "id": "prod-1",
            "title": "Widget",
            "merchant_id": "merch-1",
            "base_price_amount": 5000,
                        "base_price_currency": "NGN",
            "urls": ["https://img.example.com/a.jpg", "https://img.example.com/b.jpg"],
            "status": "active",
        }
        result = product_ui(p)
        assert result["image_url"] == "https://img.example.com/a.jpg"
        assert result["urls"] == ["https://img.example.com/a.jpg", "https://img.example.com/b.jpg"]
