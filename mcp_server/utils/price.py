"""Shared price mapping utilities for tool DTOs."""

from __future__ import annotations

from schemas.catalog import PriceView


def _map_price(amount: float | None, currency: str | None) -> PriceView:
    """Map backend price fields to canonical PriceView.

    This is the shared utility used across all tool modules to ensure
    consistent price mapping. Previously duplicated in each tool file.
    """
    return PriceView(amount=float(amount or 0), currency=currency or "NGN")
