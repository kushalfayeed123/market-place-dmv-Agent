"""Collector for published knowledge documents (policies, FAQs, guides).

Fetches ``GET /knowledge/documents`` from the backend — the admin-curated
source of truth. Only published docs are exposed by that endpoint, so this
collector never sees drafts.
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
        url = f"{self._base}{self._prefix}/knowledge/documents"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                import json

                data = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001 — collector must never crash the sync
            print(f"[kb-sync] documents source unavailable: {exc}")
            return []
        items = data.get("items", data) if isinstance(data, dict) else data
        return [d for d in items if isinstance(d, dict)] if isinstance(items, list) else []

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
