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
    # Backend LedgerBalanceResponse uses `available_balance` and `held_balance`
    # (minor-unit ints) plus a top-level `currency` and `calculated_at`.
    currency = b.get("currency", "NGN")
    return BalanceView(
        merchant_id=str(b.get("merchant_id", "")),
        merchant_name=b.get("merchant_name", ""),
        available=_map_price(b.get("available_balance"), currency),
        pending=_map_price(b.get("held_balance"), currency),
        currency=currency,
        updated_at=b.get("calculated_at", b.get("updated_at")),
    )


def _map_ledger_entry(e: dict) -> LedgerEntryView:
    # Backend LedgerEntryResponse: amount (int/minor), currency, entry_type,
    # direction, created_at, metadata (dict), payment_transaction_id, order_id
    currency = e.get("currency", "NGN")
    # Derive a human-readable description from metadata or entry_type
    desc = e.get("description")
    if not desc:
        md = e.get("metadata") or {}
        if isinstance(md, dict):
            desc = md.get("description") or md.get("reference") or md.get("note")
    if not desc:
        desc = e.get("entry_type", "ledger entry")
    return LedgerEntryView(
        id=str(e.get("id", "")),
        entry_type=e.get("entry_type", e.get("type", "unknown")),
        amount=_map_price(e.get("amount"), currency),
        balance_after=_map_price(e.get("balance_after"), currency)
        if e.get("balance_after") is not None
        else None,
        order_id=(str(e.get("order_id", "") or "") or None) if e.get("order_id") else None,
        payment_id=(str(e.get("payment_id", "") or "") or None)
        if e.get("payment_id")
        else (str(e.get("payment_transaction_id", "") or "") or None)
        if e.get("payment_transaction_id")
        else None,
        description=desc,
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
