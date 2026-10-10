"""Shared price mapping utilities for tool DTOs."""

from __future__ import annotations

from mcp_server.money import format_money
from schemas.catalog import PriceView


def _map_price(amount: float | int | None, currency: str | None) -> PriceView:
    """Map backend price fields to canonical PriceView.

    ``amount`` is stored as an **integer** in minor units (kobo / cents).
    The ``display`` field provides the pre-formatted string so the model
    never needs to interpret the raw integer.
    """
    minor = int(amount or 0)
    cur = currency or "NGN"
    return PriceView(
        amount=minor,
        currency=cur,
        display=format_money(minor, cur),
    )
