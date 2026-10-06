"""Collector for published knowledge documents (policies, FAQs, guides).

Fetches ``GET /knowledge/documents/published`` from the backend — the
customer-facing, published-only endpoint (same trust level as the public
catalog). It requires no auth, so the sync runs without a user token.
"""

from __future__ import annotations

import urllib.request
from typing import AsyncIterator

from .base import SourceDocument, scrub_untrusted


class DocumentsCollector:
    """Collects published knowledge_documents from the backend."""

    def __init__(self, base_url: str, api_prefix: str = "/api/v1", timeout: int = 15):
        self._base = base_url.rstrip("/")
        self._prefix = api_prefix
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "documents"

    def _fetch(self) -> list[dict]:
        import json

        items: list[dict] = []
        skip, limit = 0, 100  # backend caps limit at 200
        while True:
            url = f"{self._base}{self._prefix}/knowledge/published?skip={skip}&limit={limit}"
            try:
                with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                    page = json.loads(resp.read().decode())
            except Exception as exc:  # noqa: BLE001
                print(f"[kb-sync] documents source unavailable ({url}): {exc}")
                break
            if isinstance(page, dict):
                page = page.get("items", [])
            batch = [d for d in page if isinstance(d, dict)]
            items.extend(batch)
            if len(batch) < limit:
                break
            skip += limit
        return items

    async def collect(self) -> AsyncIterator[SourceDocument]:
        import asyncio

        docs = await asyncio.to_thread(self._fetch)
        for d in docs:
            title = str(d.get("title", "")).strip()
            body = str(d.get("body", "")).strip()
            if not body:
                continue
            yield SourceDocument(
                source=self.name,
                doc_id=str(d.get("id", title)),
                text=body,
                title=title,
                kind=str(d.get("doc_type", "policy")),
                topic=str(d.get("topic", "")),
                audience=str(d.get("audience", "all")),
                version=int(d.get("version", 1)),
                metadata={"origin": "knowledge_documents"},
            )
