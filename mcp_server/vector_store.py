"""RAG vector store (clean implementation; this file was recreated)."""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any, Protocol

_VS_CLEAN = True

class Embedder(Protocol):
    @property
    def dim(self) -> int:
        ...

    async def embed(self, text: str) -> list[float]:
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class TfidfEmbedder:
    def __init__(self, dim: int = 256, max_vocab: int = 10000):
        self._dim = dim
        self._max_vocab = max_vocab
        self._vocabulary: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._fitted = False

    @property
    def dim(self) -> int:
        return self._dim

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, documents: list[str]) -> None:
        df: dict[str, int] = {}
        for doc in documents:
            tokens = self._tokenize(doc)
            seen = set(tokens)
            for token in seen:
                df[token] = df.get(token, 0) + 1
        n = len(documents)
        # For small corpora keep singleton terms so rare queries still match.
        min_df = 1 if n < 50 else max(2, int(n * 0.01))
        max_df = int(n * 0.95)
        filtered = {t: c for t, c in df.items() if min_df <= c <= max_df}
        sorted_terms = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        top_terms = sorted_terms[: self._max_vocab]
        self._vocabulary = {term: idx for idx, (term, _) in enumerate(top_terms)}
        self._idf = {term: math.log((n + 1) / (count + 1)) + 1 for term, count in top_terms}
        self._fitted = True

    def _vectorize(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self._dim
        tf: dict[str, float] = {}
        for token in tokens:
            if token in self._vocabulary:
                tf[token] = tf.get(token, 0) + 1
        vector = [0.0] * self._dim
        max_tf = max(tf.values()) if tf else 1.0
        for term, count in tf.items():
            idx = self._vocabulary[term]
            if idx < self._dim:
                normalized_tf = 0.5 + 0.5 * (count / max_tf)
                vector[idx] = normalized_tf * self._idf.get(term, 1.0)
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    async def embed(self, text: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fitted")
        return self._vectorize(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fitted")
        return [self._vectorize(t) for t in texts]


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self._model = model
        self._api_key = api_key or ""
        self._dim = 1536

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_batch_sync, texts)

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        import urllib.request

        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": self._model, "input": texts}).encode(),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
        return [item["embedding"] for item in payload["data"]]


class VectorStore:
    def __init__(self, redis_client: Any, embedder: Embedder, namespace: str = "agent"):
        self._redis = redis_client
        self._embedder = embedder
        self._ns = namespace
        self._embedder_fitted = False

    def _key(self, doc_id: str) -> str:
        return f"{self._ns}:vector:{doc_id}"

    def _meta_key(self) -> str:
        return f"{self._ns}:vector:meta"

    async def upsert_many(self, docs: list[dict[str, Any]]) -> int:
        """Bulk-index documents. Each doc: {id, text, metadata?}. Returns count."""
        texts = [d["text"] for d in docs]
        embeddings = await self._embedder.embed_batch(texts)
        pipe = self._redis.pipeline()
        for doc, embedding in zip(docs, embeddings):
            data = {
                "text": doc["text"],
                "embedding": json.dumps(embedding),
                "metadata": json.dumps(doc.get("metadata", {})),
            }
            pipe.hset(self._key(doc["id"]), mapping=data)
            pipe.sadd(self._meta_key(), doc["id"])
        await pipe.execute()
        self._embedder_fitted = True
        return len(docs)

    async def get_all_texts(self) -> list[str]:
        """Return raw texts of all indexed docs (used to fit TF-IDF)."""
        doc_ids = await self._redis.smembers(self._meta_key())
        if not doc_ids:
            return []
        pipe = self._redis.pipeline()
        for doc_id in doc_ids:
            pipe.hget(self._key(doc_id), "text")
        results = await pipe.execute()
        return [t for t in results if t]

    async def upsert(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        embedding = await self._embedder.embed(text)
        data = {"text": text, "embedding": json.dumps(embedding), "metadata": json.dumps(metadata or {})}
        await self._redis.hset(self._key(doc_id), mapping=data)
        await self._redis.sadd(self._meta_key(), doc_id)

    async def delete(self, doc_id: str) -> None:
        await self._redis.delete(self._key(doc_id))
        await self._redis.srem(self._meta_key(), doc_id)

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        data = await self._redis.hgetall(self._key(doc_id))
        if not data:
            return None
        return {"id": doc_id, "text": data.get("text", ""), "metadata": json.loads(data.get("metadata", "{}"))}

    async def fit_embedder(self) -> int:
        """Fit an unfitted TF-IDF embedder from all stored document texts.

        Must be called after ingestion and before search when using
        TfidfEmbedder. Returns the number of documents used for fitting.
        """
        texts = await self.get_all_texts()
        if hasattr(self._embedder, "fit"):
            self._embedder.fit(texts)
        self._embedder_fitted = True
        return len(texts)

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._embedder_fitted and hasattr(self._embedder, "_fitted"):
            await self.fit_embedder()
        query_embedding = await self._embedder.embed(query)
        return await self.search_by_embedding(query_embedding, top_k)

    async def search_by_embedding(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        doc_ids = await self._redis.smembers(self._meta_key())
        if not doc_ids:
            return []
        pipe = self._redis.pipeline()
        for doc_id in doc_ids:
            pipe.hgetall(self._key(doc_id))
        results = await pipe.execute()
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc_id, data in zip(doc_ids, results):
            if not data:
                continue
            embedding = json.loads(data.get("embedding", "[]"))
            if not embedding:
                continue
            score = _cosine_similarity(query_embedding, embedding)
            scored.append((score, {"id": doc_id, "text": data.get("text", ""), "metadata": json.loads(data.get("metadata", "{}")), "score": round(score, 4)}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    async def count(self) -> int:
        return await self._redis.scard(self._meta_key())

    async def clear(self) -> int:
        doc_ids = await self._redis.smembers(self._meta_key())
        if doc_ids:
            pipe = self._redis.pipeline()
            for doc_id in doc_ids:
                pipe.delete(self._key(doc_id))
            pipe.delete(self._meta_key())
            await pipe.execute()
        return len(doc_ids)

    async def health_check(self) -> bool:
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def create_embedder(provider: str = "tfidf", api_key: str | None = None, model: str | None = None) -> Embedder:
    provider = provider.lower()
    if provider == "openai":
        return OpenAIEmbedder(model=model or "text-embedding-3-small", api_key=api_key)
    if provider == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedder provider: {provider!r}")
