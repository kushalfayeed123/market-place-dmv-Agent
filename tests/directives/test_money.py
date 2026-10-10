"""Unit tests for the canonical money helpers."""

import pytest
from mcp_server.money import format_money, to_minor


class TestToMinor:
    def test_whole_number(self):
        assert to_minor("1008") == 100800

    def test_decimal_string(self):
        assert to_minor("1008.50") == 100850

    def test_float_input(self):
        assert to_minor(10.00) == 1000

    def test_integer_input(self):
        assert to_minor(10) == 1000

    def test_half_up_rounding(self):
        assert to_minor("0.005") == 1  # rounds up to 1 kobo
        assert to_minor("0.004") == 0  # rounds down


class TestFormatMoney:
    def test_whole_amount_strips_trailing_zeros(self):
        assert format_money(100800, "NGN") == "₦1,008"

    def test_amount_with_kobo(self):
        assert format_money(100850, "NGN") == "₦1,008.50"

    def test_amount_with_nonzero_kobo(self):
        assert format_money(100805, "NGN") == "₦1,008.05"

    def test_zero(self):
        assert format_money(0, "NGN") == "₦0"

    def test_fifty_kobo(self):
        assert format_money(50, "NGN") == "₦0.50"

    def test_large_amount(self):
        assert format_money(100000000, "NGN") == "₦1,000,000"

    def test_default_currency_when_none(self):
        assert format_money(100800, None) == "₦1,008"

    def test_usd_symbol(self):
        assert format_money(1999, "USD") == "$19.99"

    def test_eur_symbol(self):
        assert format_money(500, "EUR") == "€5"

    def test_unknown_currency_falls_back_to_code(self):
        result = format_money(100800, "XYZ")
        assert "XYZ" in result

    def test_none_amount_treated_as_zero(self):
        assert format_money(None, "NGN") == "₦0"

    def test_float_input_is_coerced_to_int(self):
        # Should not crash on float input
        result = format_money(100800.0, "NGN")
        assert result == "₦1,008"
