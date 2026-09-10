"""Auth tests — anonymous sessions get only informational tools; financial actions prompt sign-in."""

import pytest
from mcp_server.tools_catalog import tools_for_role, BY_NAME, is_mutating_financial
from schemas.components import ComponentName


class TestAnonymousSession:

    def test_anonymous_sees_only_read_only_and_cart_tools(self):
        """Anonymous users should see read-only tools and cart tools, not financial tools."""
        tools = tools_for_role(None)
        tool_names = {t.name for t in tools}
        # Should see read-only tools
        assert "search_products" in tool_names
        assert "get_product_detail" in tool_names
        assert "get_categories" in tool_names
        # Should see cart tools
        assert "view_cart" in tool_names
        assert "add_to_cart" in tool_names
        # Should NOT see financial tools
        assert "initiate_checkout" not in tool_names
        assert "process_payment" not in tool_names
        assert "request_refund" not in tool_names
        assert "request_payout" not in tool_names
        assert "review_kyc" not in tool_names

    def test_anonymous_does_not_see_user_profile(self):
        """Anonymous users should not see authenticated-only tools."""
        tools = tools_for_role(None)
        tool_names = {t.name for t in tools}
        assert "get_user_profile" not in tool_names

    def test_buyer_sees_checkout(self):
        """Buyers should see checkout."""
        tools = tools_for_role("buyer")
        tool_names = {t.name for t in tools}
        assert "initiate_checkout" in tool_names

    def test_buyer_does_not_see_payout(self):
        """Buyers should NOT see payout."""
        tools = tools_for_role("buyer")
        tool_names = {t.name for t in tools}
        assert "request_payout" not in tool_names

    def test_admin_sees_everything(self):
        """Admin should see all tools."""
        tools = tools_for_role("platform_admin")
        tool_names = {t.name for t in tools}
        assert "review_kyc" in tool_names
        assert "request_payout" in tool_names
        assert "list_merchants" in tool_names

    def test_is_mutating_financial_identifies_financial_tools(self):
        """The is_mutating_financial function must correctly identify financial tools."""
        assert is_mutating_financial("initiate_checkout") is True
        assert is_mutating_financial("process_payment") is True
        assert is_mutating_financial("request_refund") is True
        assert is_mutating_financial("review_kyc") is True
        assert is_mutating_financial("request_payout") is True
        assert is_mutating_financial("search_products") is False
        assert is_mutating_financial("view_cart") is False
        assert is_mutating_financial("get_order_status") is False
