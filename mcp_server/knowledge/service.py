"""KnowledgeService — the query side used by runtime tools.

Hybrid retrieval (vector + lexical overlap), minimum-score threshold, filters,
and citations. Never raises: errors surface as ``kb_unavailable`` results so
the agent turn loop can fall back gracefully.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .config import KnowledgeConfig
from .embedders import Embedder


@dataclass
class KBResult:
    """One retrieval result with provenance for citation."""

    doc_id: str
    title: str
    text: str
    kind: str
    topic: str
    version: int
    score: float
    id: str = ""
    metadata: dict = field(default_factory=dict)

    def citation(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "version": self.version, "kind": self.kind}


@dataclass
class KBAnswer:
    """The answer returned to the agent for a KB query."""

    status: str                      # "ok" | "no_match" | "kb_unavailable"
    results: list[KBResult] = field(default_factory=list)
    top_score: float = 0.0
    query: str = ""
    latency_ms: int = 0
    message: str = ""


def _lexical_overlap(query: str, text: str) -> float:
    """Simple token-overlap score in [0, 1] — the keyword fallback leg."""
    q_tokens = {t for t in "".join(c.lower() if c.isalnum() else " " for c in query).split() if len(t) > 2}
    if not q_tokens:
        return 0.0
    t_tokens = {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 2}
    if not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)

class KnowledgeService:
    """Query interface over the versioned vector index."""

    def __init__(self, redis_client: Any, embedder: Embedder, config: KnowledgeConfig):
        self._redis = redis_client
        self._embedder = embedder
        self._config = config
        self._store: Any = None

    def _get_store(self) -> Any:
        if self._store is None:
            from ..vector_store import VectorStore

            self._store = VectorStore(
                redis_client=self._redis,
                embedder=_QueryShim(self._embedder),
                namespace=self._config.index_namespace,
            )
        return self._store

    async def search(
        self,
        query: str,
        *,
        topic: str | None = None,
        kind: str | None = None,
        top_k: int | None = None,
    ) -> KBAnswer:
        """Search the KB. Returns a ``KBAnswer`` — never raises to the caller."""
        started = time.monotonic()
        top_k = top_k or self._config.top_k
        try:
            store = self._get_store()
            query_vec = await self._embedder.embed(query)
            # Over-fetch so filters + lexical re-ranking still leave top_k results.
            candidates = await store.search_by_embedding(query_vec, top_k=max(top_k * 4, 20))
        except Exception as exc:  # noqa: BLE001 — typed graceful degradation
            return KBAnswer(status="kb_unavailable", query=query, message=f"KB search failed: {exc}")

        scored: list[KBResult] = []
        for cand in candidates:
            meta = cand.get("metadata", {})
            if topic and meta.get("topic") != topic:
                continue
            if kind and meta.get("kind") != kind:
                continue
            vec_score = float(cand.get("score", 0.0))
            lex_score = _lexical_overlap(query, cand.get("text", ""))
            blended = vec_score + self._config.lexical_fallback_weight * lex_score
            scored.append(KBResult(
                id=cand.get("id", ""),
                doc_id=str(meta.get("doc_id", cand.get("id", ""))),
                title=str(meta.get("title", "")),
                text=cand.get("text", ""),
                kind=str(meta.get("kind", "")),
                topic=str(meta.get("topic", "")),
                version=int(meta.get("version", 1)),
                score=round(blended, 4),
                metadata=meta,
            ))

        scored.sort(key=lambda r: r.score, reverse=True)
        top = scored[:top_k]
        top_score = top[0].score if top else 0.0
        latency_ms = int((time.monotonic() - started) * 1000)

        if not top or top_score < self._config.min_score:
            return KBAnswer(
                status="no_match",
                top_score=top_score,
                query=query,
                latency_ms=latency_ms,
                message="No confident match in the knowledge base.",
            )
        return KBAnswer(status="ok", results=top, top_score=top_score, query=query, latency_ms=latency_ms)

    async def get_document(self, doc_id: str) -> KBAnswer:
        """Fetch all chunks of one document, merged in order — for full-document citation."""
        try:
            store = self._get_store()
            doc_ids = await store._redis.smembers(store._meta_key())
        except Exception as exc:  # noqa: BLE001
            return KBAnswer(status="kb_unavailable", query=doc_id, message=f"KB unavailable: {exc}")

        chunks: list[tuple[int, KBResult]] = []
        for cid in doc_ids or []:
            cid = str(cid)
            if f":{doc_id}#chunk" not in cid:
                continue
            data = await self._redis.hgetall(f"{self._config.index_namespace}:vector:{cid}")
            if not data:
                continue
            meta = _safe_json(data.get("metadata", "{}"))
            chunks.append((int(meta.get("chunk", 0)), KBResult(
                id=cid,
                doc_id=str(meta.get("doc_id", doc_id)),
                title=str(meta.get("title", "")),
                text=data.get("text", ""),
                kind=str(meta.get("kind", "")),
                topic=str(meta.get("topic", "")),
                version=int(meta.get("version", 1)),
                score=1.0,
                metadata=meta,
            )))

        if not chunks:
            return KBAnswer(status="no_match", query=doc_id, message="Document not found in the index.")
        chunks.sort(key=lambda pair: pair[0])
        merged_text = "\n\n".join(c.text for _, c in chunks)
        first = chunks[0][1]
        merged = KBResult(
            doc_id=first.doc_id, title=first.title, text=merged_text, kind=first.kind,
            topic=first.topic, version=first.version, score=1.0, metadata=first.metadata,
        )
        return KBAnswer(status="ok", results=[merged], top_score=1.0, query=doc_id)

    async def doc_count(self) -> int:
        """Number of indexed chunks — used by health checks."""
        try:
            return await self._get_store().count()
        except Exception:  # noqa: BLE001
            return 0


class _QueryShim:
    """VectorStore-embedder adapter for query time (single embed, no fit needed)."""

    def __init__(self, embedder: Embedder):
        self._e = embedder

    @property
    def dim(self) -> int:
        return self._e.dim

    async def embed(self, text: str) -> list[float]:
        return await self._e.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._e.embed_batch(texts)


def _safe_json(raw: str) -> dict:
    import json

    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


