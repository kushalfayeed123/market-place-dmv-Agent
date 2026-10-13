"""Product search that does not silently lose results.

Why this exists. Before, `search_products` called the backend with the model's keywords
and then ran a client-side fuzzy filter over ONE page of results. That discarded valid
backend matches (the filter looked only at name, merchant and category, and the last two
are empty in production), ignored everything past the first page, and passed price filters
in the wrong unit (the model speaks naira, the database stores kobo).

Strategy, in order:
  1. backend text search (only if the backend really supports `q`, probed once and cached);
  2. a paged scan of the active catalog scored locally on title, SKU, category, merchant,
     description and attributes;
  3. when there are no keywords ('what's popular', 'under 50k'), a plain browse.
Price filters are always applied locally, converting major units (naira) to minor units.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .money import to_minor

logger = logging.getLogger("gateway.search")

FetchPage = Callable[[dict[str, Any]], Awaitable[list[dict[str, Any]]]]


@dataclass(frozen=True)
class SearchConfig:
    page_size: int = 50
    max_scan: int = 300
    max_results: int = 48
    min_match: float = 0.5

    @classmethod
    def from_env(cls) -> SearchConfig:
        return cls(
            page_size=int(os.getenv("SEARCH_PAGE_SIZE", "50")),
            max_scan=int(os.getenv("SEARCH_MAX_SCAN", "300")),
        )


@dataclass
class SearchOutcome:
    items: list[dict[str, Any]]
    total: int
    strategy: str
    query: str
    attempts: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- text

_STOPWORDS = set(
    """a an the for of with and or in on to from by at as is are be do does did have has i you we me my
    your our us it its this that these those there here please can could would should will just any some
    something anything stuff things show find search looking look want need get give list display tell
    popular trending new arrivals deals deal sale best top cheap cheapest affordable good great nice
    recommended recommend recommendation recommendations product products item items available stock
    under over below above less more than cheaper at least most within budget price prices cost
    what which who where when why how hello hi hey thanks thank ok okay yes no again also still really""".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Significant search words: lower-cased, stop words removed, plurals folded."""
    out: list[str] = []
    for raw in _WORD.findall((text or "").lower()):
        if raw in _STOPWORDS or (len(raw) < 2 and not raw.isdigit()):
            continue
        word = _stem(raw)
        if word not in out:
            out.append(word)
    return out


_NUM = r"(?:\u20a6|ngn|n|\$|usd)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([km])?"
_MAX_RE = re.compile(
    r"\b(?:under|below|less than|cheaper than|max(?:imum)?(?: of)?|up to|within|not more than)\s*" + _NUM, re.I
)
_MIN_RE = re.compile(r"\b(?:over|above|more than|at least|min(?:imum)?(?: of)?)\s*" + _NUM, re.I)


def _to_number(digits: str, suffix: str | None) -> float:
    value = float(digits.replace(",", ""))
    return value * {"k": 1e3, "m": 1e6}.get((suffix or "").lower(), 1.0)


def extract_price_hint(query: str) -> tuple[float | None, float | None, str]:
    """Pull 'under 50k' / 'over 20,000' out of a phrase. Returns (min, max, cleaned) in MAJOR units."""
    cleaned = query or ""
    price_max = price_min = None
    m = _MAX_RE.search(cleaned)
    if m:
        price_max = _to_number(m.group(1), m.group(2))
        cleaned = cleaned[: m.start()] + " " + cleaned[m.end():]
    m = _MIN_RE.search(cleaned)
    if m:
        price_min = _to_number(m.group(1), m.group(2))
        cleaned = cleaned[: m.start()] + " " + cleaned[m.end():]
    return price_min, price_max, cleaned.strip()


def _fields(p: dict[str, Any]) -> list[tuple[str, float]]:
    variants = p.get("variants") if isinstance(p.get("variants"), list) else []
    skus = " ".join(str(v.get("sku", "")) for v in variants if isinstance(v, dict))
    attrs = p.get("attributes") if isinstance(p.get("attributes"), dict) else {}
    attr_text = " ".join(str(v) for v in attrs.values())
    return [
        (str(p.get("title") or p.get("name") or "").lower(), 3.0),
        ((str(p.get("sku") or "") + " " + skus).lower(), 2.0),
        (" ".join(str(p.get(k) or "") for k in ("category_name", "merchant_name", "store_name")).lower(), 1.5),
        ((str(p.get("description") or "") + " " + attr_text).lower(), 1.0),
    ]


def _token_in(token: str, text: str) -> bool:
    if not text:
        return False
    if token in text:
        return True
    if len(token) >= 4:  # tolerate typos: 'keybord' ~ 'keyboard'
        return any(SequenceMatcher(None, token, w).ratio() >= 0.82 for w in _WORD.findall(text))
    return False


def lexical_score(tokens: list[str], product: dict[str, Any]) -> tuple[float, float]:
    """(coverage, quality). coverage = share of keywords found anywhere; quality favours title hits."""
    if not tokens:
        return (0.0, 0.0)
    fields = _fields(product)
    matched = 0
    quality = 0.0
    for tok in tokens:
        best = 0.0
        for text, weight in fields:
            if _token_in(tok, text):
                best = max(best, weight)
        if best:
            matched += 1
            quality += best / 3.0
    return (matched / len(tokens), quality / len(tokens))


# --------------------------------------------------------------------------- backend capability probe


class QSupportProbe:
    """Detects once whether the backend really filters on `q` (a nonsense query must return nothing)."""

    def __init__(self, ttl: float = 600.0) -> None:
        self._ttl = ttl
        self._value: bool | None = None
        self._at = 0.0

    async def supports(self, fetch_page: FetchPage) -> bool:
        if self._value is not None and time.monotonic() - self._at < self._ttl:
            return self._value
        try:
            items = await fetch_page({"q": "zzqxjv-no-such-product-9f3a", "skip": 0, "limit": 5})
        except Exception:  # do not cache a failure; assume the backend works
            return True
        self._value = len(items) == 0
        self._at = time.monotonic()
        if not self._value:
            logger.warning("backend ignores the 'q' parameter; searching by local scan instead")
        return self._value


_PROBE = QSupportProbe()


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for p in items:
        key = str(p.get("id", id(p)))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


# --------------------------------------------------------------------------- main entry


async def search_catalog(
    fetch_page: FetchPage,
    *,
    query: str | None,
    category_id: str | None = None,
    merchant_id: str | None = None,
    price_min: Any = None,
    price_max: Any = None,
    currency: str | None = None,
    limit: Any = 20,
    include_inactive: bool = False,
    cfg: SearchConfig | None = None,
    probe: QSupportProbe | None = None,
) -> SearchOutcome:
    cfg = cfg or SearchConfig.from_env()
    probe = probe or _PROBE
    try:
        limit_n = max(1, min(int(limit or 20), cfg.max_results))
    except (TypeError, ValueError):
        limit_n = 20

    original = (query or "").strip()
    hint_min, hint_max, cleaned = extract_price_hint(original)
    min_minor = to_minor(price_min) if price_min not in (None, "") else to_minor(hint_min)
    max_minor = to_minor(price_max) if price_max not in (None, "") else to_minor(hint_max)
    has_price = min_minor is not None or max_minor is not None
    cur = (currency or "NGN").upper()
    tokens = tokenize(cleaned)

    base: dict[str, Any] = {}
    if category_id:
        base["category_id"] = category_id
    if merchant_id:
        base["merchant_id"] = merchant_id
    attempts: list[dict[str, Any]] = []

    def keep(p: dict[str, Any]) -> bool:
        if not include_inactive and str(p.get("status") or "active").lower() != "active":
            return False
        if has_price:
            amount = p.get("base_price_amount")
            if amount is None:
                return True
            if str(p.get("base_price_currency") or cur).upper() != cur:
                return False
            amount = float(amount)
            if min_minor is not None and amount < min_minor:
                return False
            if max_minor is not None and amount > max_minor:
                return False
        return True

    async def paged(extra: dict[str, Any], max_items: int, label: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        skip = 0
        while len(out) < max_items:
            t0 = time.monotonic()
            page = await fetch_page({**base, **extra, "skip": skip, "limit": cfg.page_size})
            attempts.append({"step": label, "skip": skip, "returned": len(page), "ms": int((time.monotonic() - t0) * 1000)})
            seen_ids = {str(x.get("id")) for x in out}
            fresh = [x for x in page if str(x.get("id")) not in seen_ids]
            out.extend(fresh)
            # Stop on an empty page, a short page, or a page with nothing new (backend ignoring `skip`).
            # 'Short' is judged against 20 because some backends cap `limit` below our page size.
            if not page or not fresh or len(page) < min(cfg.page_size, 20):
                break
            skip += len(page)
        return out

    candidates: list[dict[str, Any]] = []
    strategy = "browse"

    if tokens:
        if await probe.supports(fetch_page):
            raw = await paged({"q": cleaned}, cfg.max_results * 2, "backend_q")
            candidates = _dedupe([p for p in raw if keep(p)])
            strategy = "backend"
        if not candidates:
            raw = await paged({}, cfg.max_scan, "scan")
            pool = _dedupe([p for p in raw if keep(p)])
            scored = [(lexical_score(tokens, p), p) for p in pool]
            strong = [sp for sp in scored if sp[0][0] >= cfg.min_match]
            strategy = "scan"
            if not strong:
                strong = [sp for sp in scored if sp[0][0] > 0]
                strategy = "scan_partial"
            strong.sort(key=lambda sp: sp[0], reverse=True)
            candidates = [p for _, p in strong]
        else:
            ranked = sorted(((lexical_score(tokens, p), p) for p in candidates), key=lambda sp: sp[0], reverse=True)
            candidates = [p for _, p in ranked]
    else:
        raw = await paged({}, cfg.max_scan if has_price else max(limit_n, cfg.page_size), "browse")
        candidates = _dedupe([p for p in raw if keep(p)])

    outcome = SearchOutcome(items=candidates[:limit_n], total=len(candidates), strategy=strategy, query=original, attempts=attempts)
    logger.info(
        "catalog search q=%r tokens=%s price=[%s,%s] strategy=%s found=%d steps=%s",
        original, tokens, min_minor, max_minor, strategy, len(candidates), attempts,
    )
    return outcome
