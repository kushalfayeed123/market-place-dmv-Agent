"""Embedder port + implementations.

The ``Embedder`` protocol is the seam between the KB pipeline and any
embedding provider. Implementations:

- ``NvidiaEmbedder``  — NVIDIA NIM embeddings API (production default).
- ``TfidfEmbedder``   — local TF-IDF fallback (dev/test only).
- ``FakeEmbedder``    — deterministic hash-based embedder for tests (no network).

All embedders return L2-normalised vectors so cosine similarity reduces to a dot product.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Any, Protocol


class Embedder(Protocol):
    """Port every embedding provider must satisfy."""

    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class NvidiaEmbedder:
    """NVIDIA NIM embeddings via the OpenAI-compatible ``/embeddings`` endpoint.

    Batches inputs and retries transient failures with exponential backoff.
    """

    NIM_BASE_URL = "https://integrate.api.nvidia.com/v1/embeddings"

    def __init__(
        self,
        model: str = "nvidia/nv-embedqa-e5-v5",
        api_key: str = "",
        dim: int = 1024,
        batch_size: int = 64,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self._model = model
        self._api_key = api_key
        self._dim = dim
        self._batch_size = max(1, batch_size)
        self._timeout = timeout
        self._max_retries = max_retries

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            out.extend(await self._embed_request(batch))
        return out

    async def _embed_request(self, batch: list[str]) -> list[list[float]]:
        import httpx  # local import keeps httpx optional for TF-IDF-only setups

        payload = {"input": batch, "model": self._model, "input_type": "query", "encoding_format": "float", "truncate": "END"}
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json", "Accept": "application/json"}
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(self.NIM_BASE_URL, json=payload, headers=headers)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(f"NVIDIA embeddings transient failure: HTTP {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()["data"]
                ordered = sorted(data, key=lambda item: item.get("index", 0))
                vecs = [_l2_normalize(item["embedding"]) for item in ordered]
                if len(vecs) != len(batch):
                    raise RuntimeError(f"Embedding count mismatch: sent {len(batch)}, got {len(vecs)}")
                return vecs
            except Exception as exc:  # noqa: BLE001 — deliberate: retry then raise
                last_exc = exc
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"NVIDIA embedding failed after {self._max_retries} attempts: {last_exc}")


class FakeEmbedder:
    """Deterministic hash-based embedder for tests and the golden-set CI gate.

    Maps a text to a stable pseudo-embedding via token hashing — no network,
    no API key. Similar texts overlap in hashed dimensions, enough for
    recall@k assertions on a curated golden set.
    """

    def __init__(self, dim: int = 128, model_name: str = "fake-hash-v1"):
        self._dim = dim
        self._model = model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    def _vectorize(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = "".join(c.lower() if c.isalnum() else " " for c in text).split()
        for tok in tokens:
            h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "big")
            idx = h % self._dim
            vec[idx] += 1.0
            vec[(h >> 17) % self._dim] += 0.5
        return _l2_normalize(vec)

    async def embed(self, text: str) -> list[float]:
        return self._vectorize(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain cosine — inputs are expected L2-normalised, but stay defensive."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def create_embedder(
    provider: str = "nvidia",
    *,
    api_key: str = "",
    model: str | None = None,
    dim: int | None = None,
    batch_size: int = 64,
    timeout: float = 60.0,
) -> Embedder:
    """Factory: build the configured embedder.

    Raises ValueError for unknown providers (fail fast at startup, not at query time).
    """
    provider = provider.lower()
    if provider == "nvidia":
        return NvidiaEmbedder(
            model=model or "nvidia/nv-embedqa-e5-v5",
            api_key=api_key,
            dim=dim or 1024,
            batch_size=batch_size,
            timeout=timeout,
        )
    if provider == "tfidf":
        # Reuse the single existing TF-IDF implementation.
        from ..vector_store import TfidfEmbedder  # local import avoids cycle

        return _TfidfAdapter(TfidfEmbedder(dim=dim or 256))
    if provider == "fake":
        return FakeEmbedder(dim=dim or 128)
    raise ValueError(f"Unknown embedder provider: {provider!r}")


class _TfidfAdapter:
    """Adapts the legacy TfidfEmbedder (fit-required) to the Embedder port."""

    def __init__(self, inner: Any):
        self._inner = inner
        self._model = "tfidf-local"

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def model_name(self) -> str:
        return self._model

    def fit(self, documents: list[str]) -> None:
        self._inner.fit(documents)

    async def embed(self, text: str) -> list[float]:
        return await self._inner.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.embed_batch(texts)

