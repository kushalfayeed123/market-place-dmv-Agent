"""Knowledge tools - semantic search over the vector store."""

from __future__ import annotations

from typing import Any


def register_knowledge_tools(server: Any, redis_client: Any, namespace: str = "agent") -> None:
    """Register knowledge/RAG tools backed by the Redis vector store."""
    from ..vector_store import TfidfEmbedder, VectorStore

    vector_store = VectorStore(redis_client=redis_client, embedder=TfidfEmbedder(), namespace=namespace)
    @server.tool(
        name="semantic_search",
        description="Search the knowledge base semantically for products, policies, or FAQs. Returns relevant documents ranked by similarity.",
    )
    async def semantic_search(
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        results = await vector_store.search(query, top_k=top_k)
        return [{"text": r["text"], "score": r["score"], "metadata": r.get("metadata", {})} for r in results]
