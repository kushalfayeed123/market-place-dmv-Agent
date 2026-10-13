"""Knowledge tools - semantic search over the vector store.

In production the gateway wires a ``KnowledgeService`` into the agent; this
module also supports a standalone Redis vector store (used by MCP clients and
smoke tests) via the legacy ``TfidfEmbedder`` path.
"""

from __future__ import annotations

from typing import Any, Optional


def register_knowledge_tools(
    server: Any,
    redis_client: Any = None,
    namespace: str = "agent",
    knowledge_service: Optional[Any] = None,
) -> None:
    """Register knowledge/RAG tools backed by the Redis vector store.

    When a ``knowledge_service`` is provided (the intended production path from
    ``gateway/main.py``), ``semantic_search`` routes through its hybrid
    vector + lexical + fuzzy scoring.  Otherwise it falls back to the legacy
    ``TfidfEmbedder`` + ``VectorStore`` for standalone / test use.
    """
    _service = knowledge_service

    @server.tool(
        name="semantic_search",
        description="Semantic search over the knowledge base for products, policies, or FAQs. Returns relevant documents ranked by similarity.",
    )
    async def semantic_search(
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        if _service is not None:
            kb = await _service.search(query, top_k=top_k)
            if kb.status == "ok":
                return [
                    {
                        "text": r.text,
                        "score": r.score,
                        "metadata": {
                            **(r.metadata or {}),
                            "doc_id": r.doc_id,
                            "title": r.title,
                            "kind": r.kind,
                            "topic": r.topic,
                        },
                    }
                    for r in kb.results
                ]
            return []

        # Legacy fallback: TfidfEmbedder + VectorStore.
        from ..vector_store import TfidfEmbedder, VectorStore

        if redis_client is None:
            return []
        vector_store = VectorStore(redis_client=redis_client, embedder=TfidfEmbedder(), namespace=namespace)
        results = await vector_store.search(query, top_k=top_k)
        return [
            {"text": r["text"], "score": r["score"], "metadata": r.get("metadata", {})}
            for r in results
        ]
