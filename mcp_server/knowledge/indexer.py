"""Indexer — embeds chunks and upserts/deletes them in the vector index.

Built on the existing ``VectorStore`` but namespaced per embedding model+dim
(see ``KnowledgeConfig.index_namespace``) so a model change builds a new index
and the old one is dropped only after the swap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chunking import chunk_document
from .collectors.base import SourceDocument
from .config import KnowledgeConfig
from .embedders import Embedder, cosine_similarity
from .sync_state import SyncState


@dataclass
class SyncStats:
    sources_seen: int = 0
    docs_changed: int = 0
    docs_unchanged: int = 0
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    errors: int = 0
    details: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sources_seen": self.sources_seen,
            "docs_changed": self.docs_changed,
            "docs_unchanged": self.docs_unchanged,
            "chunks_upserted": self.chunks_upserted,
            "chunks_deleted": self.chunks_deleted,
            "errors": self.errors,
        }


class KBIndexer:
    """Writes/updates the vector index for collected source documents."""

    def __init__(
        self,
        redis_client: Any,
        embedder: Embedder,
        config: KnowledgeConfig,
        sync_state: SyncState | None = None,
    ):
        self._redis = redis_client
        self._embedder = embedder
        self._config = config
        self._state = sync_state or SyncState(redis_client, config.index_namespace)
        self._store: Any = None  # lazily created VectorStore on first use

    def _get_store(self) -> Any:
        if self._store is None:
            from ..vector_store import VectorStore

            self._store = VectorStore(
                redis_client=self._redis,
                embedder=_EmbedderShim(self._embedder),
                namespace=self._config.index_namespace,
            )
        return self._store

    async def sync_document(self, doc: SourceDocument, incremental: bool = True) -> SyncStats:
        """Sync one source document: skip if unchanged, else re-chunk + re-embed."""
        stats = SyncStats(sources_seen=1)
        last_checksum = await self._state.get_checksum(doc.source, doc.doc_id) if incremental else None

        if incremental and last_checksum == doc.checksum:
            stats.docs_unchanged = 1
            return stats

        store = self._get_store()
        # Delete previous chunks of this doc (by tracked ids, fall back to scan).
        old_ids = await self._state.get_indexed_ids(doc.source, doc.doc_id)
        for old_id in old_ids:
            await store.delete(old_id)
        stats.chunks_deleted += len(old_ids)

        try:
            records = chunk_document(doc, self._config.chunk_target_tokens, self._chunk_overlap())
            # TF-IDF adapters need a fit when the corpus changes; real providers don't.
            fit = getattr(self._embedder, "fit", None)
            if fit is not None:
                fit([r["text"] for r in records])
            if records:
                await store.upsert_many(records)
            await self._state.set_indexed_ids(doc.source, doc.doc_id, {r["id"] for r in records})
            await self._state.set_checksum(doc.source, doc.doc_id, doc.checksum)
            stats.docs_changed = 1
            stats.chunks_upserted = len(records)
        except Exception as exc:  # noqa: BLE001 — one bad doc must not abort the sync
            stats.errors = 1
            stats.details.append({"doc_id": doc.doc_id, "error": str(exc)})
        return stats

    def _chunk_overlap(self) -> int:
        return self._config.chunk_overlap_tokens


class _EmbedderShim:
    """Adapts an Embedder to the duck-typed embedder VectorStore expects."""

    def __init__(self, embedder: Embedder):
        self._e = embedder
        # VectorStore checks ``hasattr(embedder, "_fitted")``; absent here → treated as fitted.
        self._shim_fitted = True

    @property
    def dim(self) -> int:
        return self._e.dim

    async def embed(self, text: str) -> list[float]:
        return await self._e.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._e.embed_batch(texts)
