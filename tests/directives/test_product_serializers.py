"""Tests for product image mapping in the agent tool mappers.

Verifies that `_map_product_card` and `_map_product_detail` correctly
include the `urls` field and fall back to `urls[0]` for `image_url` when the
backend does not provide a top-level `image_url`.

Also tests that `product_llm` and `product_ui` serializers in
backend/app/schemas/catalog.py maintain money-correctness.
"""
import pytest
from schemas.catalog import product_llm, product_ui
from mcp_server.tools.catalog import _map_product_card, _map_product_detail


class TestProductCardImageMapping:
    def test_urls_passed_through(self):
        p = {
            "id": "prod-1", "title": "Widget", "merchant_id": "m-1",
            "merchant_name": "Store", "base_price_amount": 5000,
            "base_price_currency": "NGN", "status": "active",
            "urls": ["https://img.example.com/a.jpg", "https://img.example.com/b.jpg"],
        }
        result = _map_product_card(p).model_dump()
        assert result["urls"] == ["https://img.example.com/a.jpg", "https://img.example.com/b.jpg"]

    def test_image_url_from_urls_when_no_image_url(self):
        p = {
            "id": "prod-1", "title": "Widget", "merchant_id": "m-1",
            "merchant_name": "Store", "base_price_amount": 5000,
            "base_price_currency": "NGN", "status": "active",
            "urls": ["https://img.example.com/primary.jpg"],
        }
        result = _map_product_card(p).model_dump()
        assert result["image_url"] == "https://img.example.com/primary.jpg"

    def test_explicit_image_url_takes_precedence(self):
        p = {
            "id": "prod-1", "title": "Widget", "merchant_id": "m-1",
            "merchant_name": "Store", "base_price_amount": 5000,
            "base_price_currency": "NGN", "status": "active",
            "image_url": "https://img.example.com/explicit.jpg",
            "urls": ["https://img.example.com/primary.jpg"],
        }
        result = _map_product_card(p).model_dump()
        assert result["image_url"] == "https://img.example.com/explicit.jpg"

    def test_no_image_when_urls_absent(self):
        p = {
            "id": "prod-1", "title": "Widget", "merchant_id": "m-1",
            "merchant_name": "Store", "base_price_amount": 5000,
            "base_price_currency": "NGN", "status": "active",
        }
        result = _map_product_card(p).model_dump()
        assert result["image_url"] is None
        assert result["urls"] is None

    def test_detail_includes_urls(self):
        p = {
            "id": "prod-1", "title": "Widget", "merchant_id": "m-1",
            "merchant_name": "Store", "base_price_amount": 5000,
            "base_price_currency": "NGN", "status": "active",
            "urls": ["https://img.example.com/a.jpg", "https://img.example.com/b.jpg"],
            "variants": [{"id": "v1", "sku": "SKU1", "name": "Default",
                          "base_price_amount": 5000, "base_price_currency": "NGN",
                          "quantity_available": 5, "attributes": {},
                          "inventory_policy": "tracked"}],
        }
        result = _map_product_detail(p).model_dump()
        assert result["urls"] == ["https://img.example.com/a.jpg", "https://img.example.com/b.jpg"]
        assert result["image_url"] == "https://img.example.com/a.jpg"

    def test_detail_explicit_image_url_takes_precedence(self):
        p = {
            "id": "prod-1", "title": "Widget", "merchant_id": "m-1",
            "merchant_name": "Store", "base_price_amount": 5000,
            "base_price_currency": "NGN", "status": "active",
            "image_url": "https://img.example.com/explicit.jpg",
            "urls": ["https://img.example.com/primary.jpg"],
            "variants": [{"id": "v1", "sku": "SKU1", "name": "Default",
                          "base_price_amount": 5000, "base_price_currency": "NGN",
                          "quantity_available": 5, "attributes": {},
                          "inventory_policy": "tracked"}],
        }
        result = _map_product_detail(p).model_dump()
        assert result["image_url"] == "https://img.example.com/explicit.jpg"
        assert result["urls"] == ["https://img.example.com/primary.jpg"]


class TestProductSearchQueryProp:
    """Verify that search_products results use 'query' (not 'q') for the frontend."""

    def test_search_products_uses_query_key(self):
        from gateway.agent import _map_tool_result

        body = {"items": [
            {"id": "p1", "title": "Widget", "merchant_id": "m1",
             "merchant_name": "Store", "base_price_amount": 5000,
             "base_price_currency": "NGN", "status": "active",
             "urls": ["img.jpg"]},
        ]}
        args = {"q": "widget", "limit": 20}
        result = _map_tool_result("search_products", body, args)
        assert "query" in result
        assert result["query"] == "widget"
        assert "q" not in result


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
