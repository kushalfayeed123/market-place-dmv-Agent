"""Vector store port \u2014 abstract interface for embedding storage.

Defines the ``VectorStore`` Protocol that both ``RedisVectorStore`` and
``ChromaVectorStore`` satisfy, plus a ``create_vector_store()`` factory and
metadata helpers that handle the flattening/unflattening of nested metadata
dicts required by Chroma (which only accepts scalar values).
"""
from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable


# Chroma collection-name constraints: 3\u201363 chars, [a-zA-Z0-9_-],
# must start alphanumeric, cannot be "default" (reserved).
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_collection_name(raw: str) -> str:
    """Turn an arbitrary namespace string into a valid Chroma collection name."""
    cleaned = _SAFE_NAME_RE.sub("_", raw).lstrip("_") or "kb"
    cleaned = cleaned[:63]
    if len(cleaned) < 3:
        cleaned = (cleaned + "_kb_v1")[:3]
    if not cleaned[0].isalnum():
        cleaned = "k" + cleaned
    cleaned = cleaned[:63]
    if cleaned.lower() == "default":
        cleaned = cleaned + "_db"
    return cleaned


def _flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested metadata dict into Chroma-compatible scalars.

    Chroma metadata values must be str, int, float, bool, or None.
    Nested dicts/lists are JSON-stringified; None becomes an empty string.
    """
    flat: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if v is None:
            flat[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = json.dumps(v, default=str, ensure_ascii=False)
    return flat


def _unflatten_metadata(flat: dict[str, Any] | None) -> dict[str, Any]:
    """Reverse _flatten_metadata: parse JSON-stringified values back."""
    result: dict[str, Any] = {}
    for k, v in (flat or {}).items():
        if isinstance(v, str):
            stripped = v.strip()
            if stripped and stripped[0] in "[{":
                try:
                    result[k] = json.loads(stripped)
                    continue
                except (json.JSONDecodeError, TypeError):
                    pass
        result[k] = v
    return result


def _build_filter(topic: str | None = None, kind: str | None = None) -> dict[str, Any] | None:
    """Build a Chroma where-clause from optional topic and kind filters.

    Uses $and so both keys can coexist (a bare two-key where dict is
    invalid in Chroma). Returns None (match-all) when no filters apply.
    """
    clauses: list[dict[str, Any]] = []
    if topic is not None:
        clauses.append({"topic": {"$eq": topic}})
    if kind is not None:
        clauses.append({"kind": {"$eq": kind}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


@runtime_checkable
class VectorStore(Protocol):
    """Abstract vector store implemented by RedisVectorStore and ChromaVectorStore.

    Every method is async. Implementations MUST NOT block the event loop
    (wrap synchronous clients in asyncio.to_thread).
    """

    @property
    def dim(self) -> int: ...

    async def upsert_many(self, docs: list[dict[str, Any]]) -> int: ...
    async def upsert(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None: ...
    async def delete(self, doc_id: str) -> None: ...
    async def delete_many(self, doc_ids: list[str]) -> int: ...
    async def get(self, doc_id: str) -> dict[str, Any] | None: ...
    async def list_ids(self) -> list[str]: ...
    async def search_by_embedding(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]: ...
    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...
    async def count(self) -> int: ...
    async def clear(self) -> int: ...
    async def health_check(self) -> bool: ...
    async def ready(self) -> bool: ...


def create_vector_store(
    provider: str = "redis",
    *,
    redis_client: Any = None,
    embedder: Any = None,
    namespace: str = "agent",
    chroma_path: str | None = None,
    chroma_host: str | None = None,
    chroma_port: int = 8000,
    **kwargs: Any,
) -> "VectorStore":
    """Factory: build the configured vector store implementation.

    Raises ValueError for unknown providers (fail fast at startup).
    """
    provider = provider.lower()
    if provider == "chroma":
        from .chroma_store import ChromaVectorStore
        return ChromaVectorStore(
            embedder=embedder,
            namespace=namespace,
            chroma_path=chroma_path,
            chroma_host=chroma_host,
            chroma_port=chroma_port,
        )
    if provider == "redis":
        from ..vector_store import RedisVectorStore
        return RedisVectorStore(
            redis_client=redis_client,
            embedder=embedder,
            namespace=namespace,
        )
    raise ValueError(f"Unknown vector store provider: {provider!r}. Supported: redis, chroma.")

