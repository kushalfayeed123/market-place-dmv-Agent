"""Integration tests for the ChromaVectorStore backend.

Tests use Chroma EphemeralClient (in-memory) so no disk or server is needed.
Each test gets a fresh collection name to avoid cross-test interference.
"""
from __future__ import annotations

import asyncio
import tempfile

import pytest

from mcp_server.knowledge.chroma_store import ChromaVectorStore
from mcp_server.knowledge.embedders import FakeEmbedder


def _make_store(dim=64, path=None):
    """Create a ChromaVectorStore with a FakeEmbedder for testing."""
    emb = FakeEmbedder(dim=dim)
    return ChromaVectorStore(
        embedder=emb,
        namespace=f"test_kb_{dim}",
        chroma_path=path,
    )


SAMPLE_DOCS = [
    {
        "id": "doc1#chunk0",
        "text": "Hand-stitched red leather wallet for men",
        "metadata": {"doc_id": "doc1", "title": "Leather Wallet", "kind": "product", "topic": "accessories", "chunk": 0},
    },
    {
        "id": "doc2#chunk0",
        "text": "Blue cotton t-shirt, plain white, multiple sizes",
        "metadata": {"doc_id": "doc2", "title": "Cotton T-Shirt", "kind": "product", "topic": "clothing", "chunk": 0},
    },
    {
        "id": "doc1#chunk1",
        "text": "Premium genuine leather, hand-crafted in Italy",
        "metadata": {"doc_id": "doc1", "title": "Leather Wallet", "kind": "product", "topic": "accessories", "chunk": 1},
    },
]


class TestChromaVectorStoreLifecycle:
    @pytest.fixture(autouse=True)
    async def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = _make_store(dim=64, path=self.tmpdir)
        yield
        # Cleanup
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_upsert_many_returns_count(self):
        n = await self.store.upsert_many(SAMPLE_DOCS)
        assert n == 3

    async def test_ready_false_when_empty(self):
        assert await self.store.ready() is False

    async def test_ready_true_after_upsert(self):
        await self.store.upsert_many(SAMPLE_DOCS)
        assert await self.store.ready() is True

    async def test_count_matches_upserted(self):
        await self.store.upsert_many(SAMPLE_DOCS)
        assert await self.store.count() == 3

    async def test_clear_deletes_all(self):
        await self.store.upsert_many(SAMPLE_DOCS)
        assert await self.store.count() == 3
        cleared = await self.store.clear()
        assert cleared == 3
        assert await self.store.count() == 0

    async def test_delete_removes_one(self):
        await self.store.upsert_many(SAMPLE_DOCS)
        await self.store.delete("doc2#chunk0")
        assert await self.store.count() == 2
        assert await self.store.get("doc2#chunk0") is None

    async def test_delete_many(self):
        await self.store.upsert_many(SAMPLE_DOCS)
        n = await self.store.delete_many(["doc1#chunk0", "doc1#chunk1"])
        assert n == 2
        assert await self.store.count() == 1


class TestChromaVectorStoreSearch:
    @pytest.fixture(autouse=True)
    async def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = _make_store(dim=64, path=self.tmpdir)
        await self.store.upsert_many(SAMPLE_DOCS)
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_search_returns_scored_results(self):
        results = await self.store.search("leather wallet", top_k=3)
        assert len(results) == 3
        # All results should have score in [0, 1]
        for r in results:
            assert 0.0 <= r["score"] <= 1.0
        # Sorted by descending score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_search_by_embedding_returns_text_and_metadata(self):
        query_emb = [0.1] * 64
        results = await self.store.search_by_embedding(query_emb, top_k=5)
        assert len(results) > 0
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "score" in r
            assert "metadata" in r

    async def test_search_with_zero_embedding_returns_empty(self):
        results = await self.store.search_by_embedding([], top_k=5)
        assert results == []

    async def test_search_results_contain_metadata_fields(self):
        results = await self.store.search("leather", top_k=5)
        for r in results:
            meta = r["metadata"]
            assert "doc_id" in meta
            assert "kind" in meta

    async def test_search_top_k_limits_results(self):
        results = await self.store.search("leather", top_k=2)
        assert len(results) == 2


class TestChromaVectorStoreGet:
    @pytest.fixture(autouse=True)
    async def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = _make_store(dim=64, path=self.tmpdir)
        await self.store.upsert_many(SAMPLE_DOCS)
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_get_returns_document(self):
        doc = await self.store.get("doc1#chunk0")
        assert doc is not None
        assert doc["id"] == "doc1#chunk0"
        assert "leather" in doc["text"].lower()
        assert doc["metadata"]["doc_id"] == "doc1"

    async def test_get_nonexistent_returns_none(self):
        doc = await self.store.get("nonexistent#chunk0")
        assert doc is None

    async def test_get_preserves_nested_metadata(self):
        # Upsert with nested dict metadata
        await self.store.upsert(
            "nested#chunk0",
            "test document text",
            metadata={"doc_id": "nested", "variants": [{"sku": "X1", "price": 10}], "tags": ["a", "b"]},
        )
        doc = await self.store.get("nested#chunk0")
        assert doc is not None
        # Nested dict should be preserved
        assert doc["metadata"]["variants"] == [{"sku": "X1", "price": 10}]
        assert doc["metadata"]["tags"] == ["a", "b"]


class TestChromaVectorStoreListIds:
    @pytest.fixture(autouse=True)
    async def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = _make_store(dim=64, path=self.tmpdir)
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_list_ids_empty_when_no_docs(self):
        ids = await self.store.list_ids()
        assert ids == []

    async def test_list_ids_returns_all_after_upsert(self):
        await self.store.upsert_many(SAMPLE_DOCS)
        ids = await self.store.list_ids()
        assert len(ids) == 3
        assert "doc1#chunk0" in ids
        assert "doc2#chunk0" in ids
        assert "doc1#chunk1" in ids


class TestChromaVectorStoreBatching:
    @pytest.fixture(autouse=True)
    async def _setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = _make_store(dim=10, path=self.tmpdir)
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_many_docs_upsert_in_batches(self):
        docs = [
            {"id": f"doc{i}", "text": f"document number {i} about topic {i % 3}", "metadata": {"idx": i}}
            for i in range(250)
        ]
        n = await self.store.upsert_many(docs)
        assert n == 250
        assert await self.store.count() == 250


class TestChromaVectorStoreDimProperty:
    async def test_dim_matches_embedder(self):
        emb = FakeEmbedder(dim=128)
        store = ChromaVectorStore(embedder=emb, namespace="test_dim_128")
        assert store.dim == 128

    async def test_dim_matches_embedder_256(self):
        emb = FakeEmbedder(dim=256)
        store = ChromaVectorStore(embedder=emb, namespace="test_dim_256")
        assert store.dim == 256

