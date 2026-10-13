"""Tests for fuzzy string matching in the KnowledgeService scoring."""
from mcp_server.knowledge.service import (
    _fuzzy_string_score,
    _fuzzy_token_score,
    _fuzzy_score_text,
    _fuzzy_score_dict,
    _is_weak_embedder,
    _lexical_overlap,
)
from mcp_server.knowledge.embedders import FakeEmbedder
from mcp_server.knowledge.config import KnowledgeConfig


class TestFuzzyStringScore:
    def test_exact_substring_bonus(self):
        """Query that is a substring of text should score high."""
        score = _fuzzy_string_score("red", "red leather wallet handmade")
        assert score >= 0.9

    def test_exact_match(self):
        assert _fuzzy_string_score("red", "red") == 1.0

    def test_typo_tolerant(self):
        """'lether' should partially match 'leather'."""
        score = _fuzzy_string_score("lether", "leather wallet")
        assert score > 0.5

    def test_no_match(self):
        assert _fuzzy_string_score("xyz", "red leather wallet") < 0.3

    def test_empty_inputs(self):
        assert _fuzzy_token_score("", "some text") == 0.0
        assert _fuzzy_token_score("query", "") == 0.0


class TestFuzzyTokenScore:
    def test_exact_token_overlap(self):
        """Exact matching tokens should score high."""
        score = _fuzzy_token_score("red wallet", "red leather wallet")
        assert score >= 0.8

    def test_partial_token_containment(self):
        """'red' should match token 'reddish' via substring containment."""
        score = _fuzzy_token_score("red", "reddish wallet")
        assert score > 0.8

    def test_typo_token(self):
        """'lether' should partially match 'leather' token."""
        score = _fuzzy_token_score("lether", "leather wallet")
        assert score > 0.5

    def test_no_overlap(self):
        assert _fuzzy_token_score("xyz", "red leather wallet") < 0.3

    def test_empty_inputs(self):
        assert _fuzzy_token_score("", "some text") == 0.0
        assert _fuzzy_token_score("query", "") == 0.0

    def test_false_positive_token_not_matched(self):
        """Unrelated tokens with incidental character overlap should not score.

        'physics' vs 'sizes' has a SequenceMatcher ratio of ~0.5, which is
        a false positive.  The 0.6 threshold should reject it.
        """
        score = _fuzzy_token_score("quantum physics", "Cotton t-shirt plain white, multiple sizes, product_id=prod-2")
        assert score == 0.0


class TestFuzzyScoreText:
    def test_combines_lexical_and_fuzzy(self):
        """Should return the max of lexical, fuzzy-str, and fuzzy-token scores."""
        text = "Hand-stitched leather wallet with red accents"
        query = "red leather wallet"
        score = _fuzzy_score_text(query, text)
        assert score > 0.5

    def test_typo_in_query(self):
        """Typo in query ('lether') should still find products with 'leather'."""
        text = "Genuine leather wallet, handcrafted"
        score = _fuzzy_score_text("lether wallet", text)
        assert score > 0.4

    def test_short_query_matches_attribute(self):
        """A short query like 'red' should match a product with 'red' in text."""
        text = "Cotton t-shirt in red color, size medium"
        score = _fuzzy_score_text("red", text)
        assert score > 0.4


class TestFuzzyScoreDict:
    def test_scores_by_text_fields(self):
        product = {"name": "Red Leather Wallet", "merchant_name": "Leather Co", "category_name": "Accessories"}
        score = _fuzzy_score_dict("red wallet", product, ["name", "merchant_name", "category_name"])
        assert score > 0.5

    def test_scores_by_merchant_name(self):
        """Query matching merchant name should score reasonably."""
        product = {"name": "USB Cable", "merchant_name": "Leather Co", "category_name": "Accessories"}
        score = _fuzzy_score_dict("leather", product, ["name", "merchant_name", "category_name"])
        assert score > 0.4

    def test_empty_fields(self):
        product = {"name": "", "merchant_name": "", "category_name": ""}
        assert _fuzzy_score_dict("query", product, ["name", "merchant_name", "category_name"]) == 0.0

    def test_query_matches_category(self):
        """Query matching category should score reasonably."""
        product = {"name": "USB Cable", "merchant_name": "Tech Store", "category_name": "Electronics"}
        score = _fuzzy_score_dict("electronics", product, ["name", "merchant_name", "category_name"])
        assert score > 0.4


class TestIsWeakEmbedder:
    def test_fake_embedder_is_weak(self):
        emb = FakeEmbedder(dim=64)
        assert _is_weak_embedder(emb) is True

    def test_strong_embedder_not_weak(self):
        class StrongEmbedder:
            model_name = "nvidia/nv-embedqa-e5-v5"
        assert _is_weak_embedder(StrongEmbedder()) is False


class TestLexicalOverlap:
    def test_exact_tokens(self):
        assert _lexical_overlap("red wallet", "red leather wallet") == 1.0

    def test_partial_tokens(self):
        """Only matching tokens count — 'red' matches but 'wallet' doesn't."""
        score = _lexical_overlap("red wallet", "red leather bag")
        assert score == 0.5  # 1 of 2 tokens matches

    def test_no_overlap(self):
        assert _lexical_overlap("xyz", "red leather wallet") == 0.0

    def test_empty_query(self):
        assert _lexical_overlap("", "red leather wallet") == 0.0
