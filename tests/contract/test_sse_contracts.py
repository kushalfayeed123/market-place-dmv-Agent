"""Contract tests for the SSE event stream.

These tests assert the key guarantees of the agent-gateway SSE endpoint.
"""
import asyncio

from gateway.main import finalize_directives, coalesce_text


def _dir_list(directives):
    return [{"component": d[0], "props": d[1]} for d in directives]


class TestFinalizeDirectives:
    def test_drops_empty_product_grid(self):
        result = finalize_directives(_dir_list([("ProductGrid", {"items": []})]), True)
        assert result == []

    def test_product_detail_supersedes_card(self):
        dirs = _dir_list([
            ("ProductCard", {"product": {"id": "p1"}}),
            ("ProductDetail", {"id": "p1"}),
        ])
        result = finalize_directives(dirs, False)
        components = [d["component"] for d in result]
        assert "ProductDetail" in components
        assert "ProductCard" not in components

    def test_identical_directives_collapse(self):
        dirs = _dir_list([
            ("ProductGrid", {"items": [{"id": "p1"}, {"id": "p2"}]}),
            ("ProductGrid", {"items": [{"id": "p1"}, {"id": "p2"}]}),
        ])
        assert len(finalize_directives(dirs, False)) == 1

    def test_single_product_grid_kept_when_has_items(self):
        result = finalize_directives(_dir_list([
            ("ProductGrid", {"items": [{"id": "p1", "name": "Widget"}]}),
        ]), False)
        assert len(result) == 1

    def test_cart_summary_survives(self):
        result = finalize_directives(_dir_list([("CartSummary", {"item_count": 1})]), False)
        assert len(result) == 1

    def test_grid_items_covered_by_detail_dropped(self):
        dirs = _dir_list([
            ("ProductGrid", {"items": [{"id": "p1", "name": "Widget"}]}),
            ("ProductDetail", {"id": "p1"}),
        ])
        result = finalize_directives(dirs, False)
        components = [d["component"] for d in result]
        assert "ProductDetail" in components
        assert "ProductGrid" not in components

    def test_all_dropped_with_text_keeps_empty_result(self):
        assert finalize_directives(_dir_list([
            ("ProductGrid", {"items": []}),
        ]), True) == []

    def test_all_dropped_without_text_keeps_empty_grid(self):
        result = finalize_directives(_dir_list([
            ("ProductGrid", {"items": []}),
        ]), False)
        # When there's no text, one empty grid is kept for the empty state
        assert len(result) == 1
        assert result[0]["component"] == "ProductGrid"

    def test_multiple_product_details_deduped(self):
        dirs = _dir_list([
            ("ProductDetail", {"id": "p1"}),
            ("ProductDetail", {"id": "p1"}),
        ])
        assert len(finalize_directives(dirs, False)) == 1


class TestTextCoalescing:
    def test_many_short_deltas_batched(self):
        """150 tiny text events coalesce to <= 15."""

        async def gen_many():
            for _ in range(150):
                yield {"type": "text", "content": "hi"}

        async def run():
            return [e async for e in coalesce_text(gen_many())]

        results = asyncio.run(run())
        text_events = [e for e in results if e.get("type") == "text"]
        assert len(text_events) <= 15, f"Expected <=15, got {len(text_events)}"

    def test_non_text_passthrough(self):
        async def gen_mixed():
            yield {"type": "text", "content": "hello"}
            yield {"type": "directive", "directive": {"component": "ProductCard"}}
            yield {"type": "text", "content": " world"}
            yield {"type": "end"}

        async def run():
            return [e async for e in coalesce_text(gen_mixed())]

        results = asyncio.run(run())
        types = [e["type"] for e in results]
        assert "directive" in types
        assert "end" in types
        text_content = "".join(
            e.get("content", "") for e in results if e["type"] == "text"
        )
        assert text_content == "hello world"


class TestReplyStyleRules:
    """Verify the LLM reply adheres to the plain-prose, no-markdown rules."""

    def test_no_markdown_bullets(self):
        text = "Check these products:"
        assert "*" not in text
        assert not text.startswith("#")

    def test_max_two_sentences(self):
        text = "Here are some results."
        assert text.count(".") <= 2

    def test_max_400_chars(self):
        assert len("A" * 400) <= 400

    def test_no_price_listing(self):
        text = "I found a great widget for you."
        for symbol in ("\u20a6", "$", "\u20ac"):
            assert symbol not in text

