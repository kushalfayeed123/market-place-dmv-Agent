"""Collector for catalog data: products, categories (public endpoints).

Product text is UNTRUSTED — scrubbed before entering the index.
"""

from __future__ import annotations

import json
import urllib.request
from typing import AsyncIterator

from .base import SourceDocument, scrub_untrusted


class CatalogCollector:
    """Collects public catalog data for grounded product answers."""

    def __init__(self, base_url: str, api_prefix: str = "/api/v1", timeout: int = 15, product_limit: int = 200):
        self._base = base_url.rstrip("/")
        self._prefix = api_prefix
        self._timeout = timeout
        self._product_limit = product_limit

    @property
    def name(self) -> str:
        return "catalog"

    def _get(self, path: str) -> object | None:
        try:
            with urllib.request.urlopen(f"{self._base}{self._prefix}{path}", timeout=self._timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            print(f"[kb-sync] catalog {path} unavailable: {exc}")
            return None

    def _fetch(self) -> list[SourceDocument]:
        docs: list[SourceDocument] = []

        categories = self._get("/catalog/categories") or []
        cat_names: dict[str, str] = {}
        for c in categories if isinstance(categories, list) else []:
            if isinstance(c, dict) and c.get("id"):
                cat_names[str(c["id"])] = str(c.get("name", ""))
                docs.append(SourceDocument(
                    source=self.name,
                    doc_id=f"category-{c['id']}",
                    text=f"Category: {c.get('name', '')}. {c.get('description', '')}".strip(),
                    title=f"Category: {c.get('name', '')}",
                    kind="category",
                    metadata={"origin": "catalog", "category_id": str(c["id"])},
                ))

        products = self._get(f"/catalog/products?limit={self._product_limit}") or []
        items = products.get("items", products) if isinstance(products, dict) else products
        for p in items if isinstance(items, list) else []:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("id", p.get("product_id", "unknown")))
            cat = cat_names.get(str(p.get("category_id", "")), str(p.get("category", "")))
            name = str(p.get("name", ""))
            desc = scrub_untrusted(str(p.get("description", "")))
            text = " ".join(x for x in [name, desc, cat] if x).strip()
            if not text:
                continue
            docs.append(SourceDocument(
                source=self.name,
                doc_id=f"product-{pid}",
                text=text,
                title=name,
                kind="product",
                topic=cat,
                metadata={"origin": "catalog", "product_id": pid, "merchant_id": str(p.get("merchant_id", ""))},
            ))
        return docs

    async def collect(self) -> AsyncIterator[SourceDocument]:
        import asyncio

        for doc in await asyncio.to_thread(self._fetch):
            yield doc
