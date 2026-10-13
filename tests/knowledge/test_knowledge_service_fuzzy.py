"""Tests for KnowledgeService fuzzy scoring and adaptive thresholds."""
import pytest
from mcp_server.knowledge.service import KnowledgeService
from mcp_server.knowledge.config import KnowledgeConfig
from mcp_server.knowledge.embedders import FakeEmbedder


class FakeStore:
    """Minimal fake of VectorStore.search_by_embedding."""
    def __init__(self, docs):
        self._docs = docs

    @property
    def dim(self):
        return 128

    async def embed(self, text):
        return [0.0] * 128

    async def search_by_embedding(self, query_embedding, top_k=5):
        return self._docs[:top_k]

    async def count(self):
        return len(self._docs)


@pytest.fixture
def weak_service():
    """KnowledgeService with FakeEmbedder (weak, hash-based)."""
    from mcp_server.knowledge.service import _QueryShim
    config = KnowledgeConfig()
    embedder = FakeEmbedder(dim=128)
    return KnowledgeService(redis_client=None, embedder=embedder, config=config)


class TestKnowledgeServiceFuzzyScoring:
    """Verify that products are surfaced by fuzzy proximity, not direct matching."""

    @staticmethod
    def _make_docs():
        return [
            {
                "id": "catalog:prod-1#chunk0000",
                "text": "Hand-stitched genuine leather wallet for men, brown color, product_id=prod-1",
                "score": 0.02,  # near-zero cosine with FakeEmbedder
                "metadata": {"doc_id": "prod-1", "title": "Leather Wallet",
                             "kind": "product", "topic": "Accessories"},
            },
            {
                "id": "catalog:prod-2#chunk0000",
                "text": "Cotton t-shirt plain white, multiple sizes, product_id=prod-2",
                "score": 0.01,
                "metadata": {"doc_id": "prod-2", "title": "Cotton T-Shirt",
                             "kind": "product", "topic": "Clothing"},
            },
            {
                "id": "catalog:prod-3#chunk0000",
                "text": "Stainless steel water bottle 500ml, keeps drinks cold, product_id=prod-3",
                "score": 0.0,
                "metadata": {"doc_id": "prod-3", "title": "Water Bottle",
                             "kind": "product", "topic": "Accessories"},
            },
        ]

    async def test_fuzzy_match_surfaces_relevant_product(self, weak_service):
        """Query 'red lether wallet' should match 'leather wallet' by fuzzy similarity."""
        docs = self._make_docs()
        weak_service._store = FakeStore(docs)

        answer = await weak_service.search("red lether wallet", top_k=5)

        assert answer.status == "ok"
        assert len(answer.results) >= 1
        # The leather wallet should rank highest
        top = answer.results[0]
        assert "leather" in top.text.lower()
        assert top.score > 0.1

    async def test_typo_in_query_still_finds_product(self, weak_service):
        """'lether' (typo) should still match 'leather'."""
        docs = self._make_docs()
        weak_service._store = FakeStore(docs)

        answer = await weak_service.search("lether wallet", top_k=5)

        assert answer.status == "ok"
        assert len(answer.results) >= 1
        assert "leather" in answer.results[0].text.lower()

    async def test_short_query_matches_attribute(self, weak_service):
        """A short query 'brown' should match the leather wallet (brown color)."""
        docs = self._make_docs()
        weak_service._store = FakeStore(docs)

        answer = await weak_service.search("brown wallet", top_k=5)

        assert answer.status == "ok"
        top_ids = [r.doc_id for r in answer.results]
        assert "prod-1" in top_ids

    async def test_irrelevant_query_returns_no_match(self, weak_service):
        """A query with no relation to any product should not return results."""
        docs = self._make_docs()
        weak_service._store = FakeStore(docs)

        answer = await weak_service.search("quantum physics", top_k=5)

        # Should be no_match because fuzzy scores are too low
        assert answer.status == "no_match"

    async def test_min_score_adjusted_for_weak_embedder(self, weak_service):
        """With FakeEmbedder, the effective min_score should be 0.10, not 0.30."""
        assert weak_service._min_score == 0.10
        assert weak_service._vec_w == 0.0
        assert weak_service._fuzzy_w == 0.60

    async def test_ranking_by_fuzzy_score(self, weak_service):
        """Relevant products should rank above unrelated ones."""
        docs = self._make_docs()
        weak_service._store = FakeStore(docs)

        answer = await weak_service.search("leather", top_k=5)

        assert answer.status == "ok"
        # Leather wallet should be ranked above t-shirt and water bottle
        assert answer.results[0].doc_id == "prod-1"
