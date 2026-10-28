"""Unit tests for the VectorStore Protocol, factory, and metadata helpers.

Covers:
  - _sanitize_collection_name: valid Chroma collection names
  - _flatten_metadata / _unflatten_metadata: round-trip with nested data
  - _build_filter: Chroma where-clause construction
  - create_vector_store: factory dispatch for redis and chroma providers
  - VectorStore Protocol: isinstance (runtime_checkable) on both backends
"""
from __future__ import annotations

import pytest

from mcp_server.knowledge.vectors import (
    VectorStore,
    _build_filter,
    _flatten_metadata,
    _sanitize_collection_name,
    _unflatten_metadata,
    create_vector_store,
)


class TestSanitizeCollectionName:
    def test_normal_namespace_passes_through(self):
        name = _sanitize_collection_name("agent:kb:v_model_64:v1")
        assert name == "agent_kb_v_model_64_v1"

    def test_long_name_truncated_to_63_chars(self):
        long_ns = "a" * 200
        name = _sanitize_collection_name(long_ns)
        assert len(name) <= 63

    def test_special_chars_replaced_with_underscore(self):
        name = _sanitize_collection_name("test:ns.with.dot:kb:v1")
        assert "." not in name
        assert name == "test_ns_with_dot_kb_v1"

    def test_empty_string_gets_default(self):
        name = _sanitize_collection_name("")
        assert len(name) >= 3
        assert name[0].isalnum()

    def test_starts_alphanumeric(self):
        name = _sanitize_collection_name("---leading-dashes")
        assert name[0].isalnum()

    def test_reserved_default_name_suffixed(self):
        name = _sanitize_collection_name("default")
        assert name != "default"


class TestFlattenUnflattenMetadata:
    def test_flat_scalars_preserved(self):
        meta = {"title": "Wallet", "price": 48.0, "in_stock": True, "qty": 3}
        flat = _flatten_metadata(meta)
        assert flat == meta

    def test_nested_dict_json_stringified(self):
        meta = {"variants": [{"sku": "RLW-BLK", "price": 48}]}
        flat = _flatten_metadata(meta)
        assert isinstance(flat["variants"], str)
        assert "RLW-BLK" in flat["variants"]

    def test_none_becomes_empty_string(self):
        meta = {"title": None, "price": 10}
        flat = _flatten_metadata(meta)
        assert flat["title"] == ""
        assert flat["price"] == 10

    def test_round_trip_nested(self):
        original = {"doc_id": "doc-1", "variants": [{"sku": "X1", "price": 10}], "tags": ["a", "b"]}
        flat = _flatten_metadata(original)
        recovered = _unflatten_metadata(flat)
        assert recovered["doc_id"] == "doc-1"
        assert recovered["variants"] == [{"sku": "X1", "price": 10}]
        assert recovered["tags"] == ["a", "b"]

    def test_empty_dict_round_trip(self):
        assert _unflatten_metadata({}) == {}
        assert _unflatten_metadata(None) == {}

    def test_non_json_string_stays_string(self):
        result = _unflatten_metadata({"title": "Hello World"})
        assert result["title"] == "Hello World"


class TestBuildFilter:
    def test_no_filters_returns_none(self):
        assert _build_filter() is None
        assert _build_filter(None, None) is None

    def test_single_topic_filter(self):
        f = _build_filter(topic="returns")
        assert f == {"topic": {"$eq": "returns"}}

    def test_single_kind_filter(self):
        f = _build_filter(kind="product")
        assert f == {"kind": {"$eq": "product"}}

    def test_both_filters_use_and(self):
        f = _build_filter(topic="returns", kind="product")
        assert f == {"$and": [{"topic": {"$eq": "returns"}}, {"kind": {"$eq": "product"}}]}


class FakeEmbedder:
    """Minimal fake for factory tests."""
    def __init__(self, dim=64):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    async def embed(self, text, input_type=None):
        return [0.0] * self._dim

    async def embed_batch(self, texts, input_type=None):
        return [[0.0] * self._dim for _ in texts]


class TestCreateVectorStore:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown vector store provider"):
            create_vector_store(provider="mongodb", embedder=FakeEmbedder())

    def test_case_insensitive_provider(self):
        # "CHROMA" should work the same as "chroma"
        store = create_vector_store(provider="CHROMA", embedder=FakeEmbedder(), namespace="test")
        assert store.__class__.__name__ == "ChromaVectorStore"


class TestVectorStoreProtocol:
    def test_runtime_checkable_with_chroma(self):
        store = create_vector_store(provider="chroma", embedder=FakeEmbedder(), namespace="test")
        assert isinstance(store, VectorStore)

    def test_runtime_checkable_with_redis(self):
        import fakeredis.aioredis
        fake_redis = fakeredis.aioredis.FakeRedis()
        store = create_vector_store(
            provider="redis", redis_client=fake_redis, embedder=FakeEmbedder(), namespace="test"
        )
        assert isinstance(store, VectorStore)

