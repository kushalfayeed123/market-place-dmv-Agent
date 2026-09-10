"""Ledger tools — get_merchant_balance, get_merchant_ledger."""

from __future__ import annotations

from typing import Any

from mcp_server.backend_client import BackendClient
from mcp_server.utils.price import (
    _map_price,
)
from schemas.ledger import BalanceView, LedgerEntryView

# def _map_price(amount: float | None, currency: str | None) -> PriceView:
#     return PriceView(amount=float(amount or 0), currency=currency or "NGN")


def _map_balance(b: dict) -> BalanceView:
    return BalanceView(
        merchant_id=str(b.get("merchant_id", "")),
        merchant_name=b.get("merchant_name", ""),
        available=_map_price(b.get("available_amount"), b.get("currency")),
        pending=_map_price(b.get("pending_amount"), b.get("currency")),
        currency=b.get("currency", "NGN"),
        updated_at=b.get("updated_at"),
    )


def _map_ledger_entry(e: dict) -> LedgerEntryView:
    return LedgerEntryView(
        id=str(e.get("id", "")),
        entry_type=e.get("entry_type", ""),
        amount=_map_price(e.get("amount"), e.get("currency")),
        balance_after=_map_price(e.get("balance_after_amount"), e.get("currency"))
        if e.get("balance_after_amount") is not None
        else None,
        order_id=str(e.get("order_id", "") or "") or None,
        payment_id=str(e.get("payment_id", "") or "") or None,
        description=e.get("description"),
        created_at=e.get("created_at", ""),
        entry_group_id=e.get("entry_group_id"),
    )


def register_ledger_tools(server: Any, client: BackendClient) -> None:
    """Register ledger tools."""

    @server.tool(
        name="get_merchant_balance",
        description="Get merchant balance. Read-only, merchant-scoped.",
    )
    async def get_merchant_balance(
        merchant_id: str,
        token: str | None = None,
    ) -> dict:
        """Get the balance for a merchant."""
        body = await client.call(
            "GET",
            "/ledger/balance/{merchant_id}",
            token=token,
            path_params={"merchant_id": merchant_id},
        )
        return _map_balance(body).model_dump()

    @server.tool(
        name="get_merchant_ledger",
        description="Get merchant ledger entries. Read-only, merchant-scoped.",
    )
    async def get_merchant_ledger(
        merchant_id: str | None = None,
        entry_type: str | None = None,
        skip: int = 0,
        limit: int = 50,
        token: str | None = None,
    ) -> list[dict]:
        """Get ledger entries, optionally filtered."""
        params: dict[str, Any] = {"skip": skip, "limit": limit}
        if merchant_id:
            params["merchant_id"] = merchant_id
        if entry_type:
            params["entry_type"] = entry_type
        body = await client.call("GET", "/ledger/entries", token=token, params=params)
        items = body.get("items", body) if isinstance(body, dict) else body
        if not isinstance(items, list):
            items = [items] if isinstance(items, dict) else []
        return [_map_ledger_entry(e).model_dump() for e in items]
