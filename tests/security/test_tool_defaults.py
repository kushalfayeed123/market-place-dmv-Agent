"""Tests for _apply_tool_defaults — ensures default parameter values from
the TOOL_PARAMETERS schema are injected into the backend request body.

This prevents 422 "Field required" errors when the LLM omits a parameter
that has a default in the schema (e.g. ``process_payment.provider``).
"""

import pytest

from gateway.agent import _apply_tool_defaults
from mcp_server.tools_catalog import TOOL_PARAMETERS


class TestApplyToolDefaults:
    """Verify that default values are applied correctly."""

    def test_process_payment_provider_default(self):
        """When provider is omitted, the default 'paystack' must be applied."""
        args: dict = {"order_id": "abc-123"}
        _apply_tool_defaults("process_payment", args)
        assert args["provider"] == "paystack"

    def test_process_payment_provider_not_overridden(self):
        """When provider is explicitly provided, it must NOT be overridden."""
        args: dict = {"order_id": "abc-123", "provider": "stripe"}
        _apply_tool_defaults("process_payment", args)
        assert args["provider"] == "stripe"

    def test_defaults_do_not_add_missing_undefined_keys(self):
        """Only keys with an explicit default in the schema should be added."""
        args: dict = {"order_id": "abc-123"}
        _apply_tool_defaults("process_payment", args)
        # 'order_id' was already there; no extra unexpected keys added
        assert set(args.keys()) == {"order_id", "provider"}

    @pytest.mark.parametrize("tool_name", sorted(TOOL_PARAMETERS.keys()))
    def test_no_tool_has_undefined_default(self, tool_name):
        """Every default in TOOL_PARAMETERS must be a concrete value."""
        schema = TOOL_PARAMETERS[tool_name]
        for prop_name, prop in schema.get("properties", {}).items():
            if "default" in prop:
                assert prop["default"] is not None, (
                    f"Tool {tool_name} property {prop_name} has a None default"
                )

    def test_unknown_tool_is_noop(self):
        """Applying defaults for an unknown tool must not crash or add keys."""
        args: dict = {"foo": "bar"}
        _apply_tool_defaults("nonexistent_tool", args)
        assert args == {"foo": "bar"}

    def test_request_payout_currency_default(self):
        """request_payout.currency should default to NGN when omitted."""
        args: dict = {"merchant_id": "m1", "amount": 5000}
        _apply_tool_defaults("request_payout", args)
        assert args["currency"] == "NGN"

    def test_get_merchant_ledger_limit_default(self):
        """get_merchant_ledger.limit should default to 100 when omitted."""
        args: dict = {"merchant_id": "m1"}
        _apply_tool_defaults("get_merchant_ledger", args)
        assert args["skip"] == 0
        assert args["limit"] == 100
