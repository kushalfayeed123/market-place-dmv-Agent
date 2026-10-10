"""Canonical money helpers — single source of truth for minor-unit math and display.

All monetary amounts stored in the database and exchanged over the REST / MCP
boundary are integers in the *smallest* currency unit (kobo for NGN, cents for
USD).  The agent layer (model context) must **never** see these raw integers
— it should only ever use the pre-formatted ``display`` strings produced by
``format_money``.

This module is the shared implementation used by:
  - ``Agent/mcp_server/utils/price.py``    (PriceView.display)
  - ``Agent/mcp_server/tools/orders.py``    (order price view)
  - ``backend/app/schemas/catalog.py``      (product_llm / product_ui)
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

# ISO 4217 → common-symbol lookup.  Unknown currencies fall back to the code.
_CURRENCY_SYMBOLS: dict[str, str] = {
    "NGN": "₦",
    "USD": "$",
    "GBP": "£",
    "EUR": "€",
}


def to_minor(major: float | int | str | Decimal) -> int:
    """Convert a major-unit amount to integer minor units.

    >>> to_minor("1008.50")
    100850
    >>> to_minor(10)
    1000
    """
    d = Decimal(str(major))
    return int((d * 100).to_integral_value(rounding=ROUND_HALF_UP))


def format_money(amount_minor: int | float | None, currency: str | None) -> str:
    """Format a minor-unit amount as a human-readable display string.

    Trailing ``.00`` is stripped so whole amounts render without decimals,
    while fractional amounts keep their cents:

    >>> format_money(100800, "NGN")
    '₦1,008'
    >>> format_money(100850, "NGN")
    '₦1,008.50'
    >>> format_money(0, "NGN")
    '₦0'
    """
    amount = int(amount_minor or 0)
    cur = (currency or "NGN").upper()
    major = Decimal(amount) / Decimal(100)
    symbol = _CURRENCY_SYMBOLS.get(cur, cur)
    # Always format with 2 dp, then strip trailing ".00" for whole amounts.
    s = f"{major:,.2f}"
    if s.endswith(".00"):
        s = s[:-3]
    return f"{symbol}{s}"
