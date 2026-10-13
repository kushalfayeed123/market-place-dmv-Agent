"""Money helpers.

Every stored amount in this system is an integer in MINOR units (kobo for NGN):
100800 means NGN 1,008. The model must never see raw minor-unit amounts, because
it will read 100800 as NGN 100,800. `llm_view` swaps every price object for a
display string before a tool result is shown to the model.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

_SYMBOLS = {"NGN": "\u20a6", "USD": "$", "GBP": "\u00a3", "EUR": "\u20ac"}


def to_minor(major: Any) -> int | None:
    """Convert a major-unit amount (naira) to minor units (kobo). None if unparseable."""
    if major is None or major == "":
        return None
    try:
        return int((Decimal(str(major)) * 100).quantize(Decimal("1")))
    except (InvalidOperation, ValueError):
        return None


def format_money(amount_minor: Any, currency: str | None) -> str:
    """Format a minor-unit amount for people: 100800 NGN -> '₦1,008'."""
    try:
        major = Decimal(str(amount_minor)) / 100
    except (InvalidOperation, ValueError):
        return ""
    cur = (currency or "NGN").upper()
    symbol = _SYMBOLS.get(cur, cur + " ")
    if major == major.to_integral_value():
        return f"{symbol}{major:,.0f}"
    return f"{symbol}{major:,.2f}"


# Fields the model does not need (they only cost tokens).
_DROP_FOR_LLM = {"urls", "image_url", "sources", "created_at", "updated_at"}


def _is_price(d: dict) -> bool:
    return "amount" in d and "currency" in d and set(d.keys()) <= {"amount", "currency"}


def llm_view(data: Any) -> Any:
    """Return a copy of a tool result that is safe to show to the model.

    Price objects become {'display': '₦1,008', 'currency': 'NGN'}; heavy, useless
    fields are dropped. The UI keeps receiving the original minor-unit data.
    """
    if isinstance(data, dict):
        if _is_price(data):
            return {"display": format_money(data["amount"], data["currency"]), "currency": data["currency"]}
        return {k: llm_view(v) for k, v in data.items() if k not in _DROP_FOR_LLM}
    if isinstance(data, list):
        return [llm_view(x) for x in data]
    return data
