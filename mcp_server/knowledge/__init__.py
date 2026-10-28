"""Knowledge base pipeline for the agent layer.

Public surface:
- ``KnowledgeConfig`` / ``load_knowledge_config`` — configuration
- ``create_embedder`` / ``Embedder`` — embedding port
- ``KnowledgeService`` — query-side service used by the runtime tools
- ``run_sync`` — ingestion orchestration (CLI + scheduled task)
"""

from .config import KnowledgeConfig, load_knowledge_config
from .vectors import VectorStore, create_vector_store
from .chroma_store import ChromaVectorStore
from .embedders import (
    Embedder,
    FakeEmbedder,
    NvidiaEmbedder,
    cosine_similarity,
    create_embedder,
)

__all__ = [
    "KnowledgeConfig",
    "load_knowledge_config",
    "Embedder",
    "FakeEmbedder",
    "NvidiaEmbedder",
    "cosine_similarity",
    "create_embedder",
    "VectorStore",
    "create_vector_store",
    "ChromaVectorStore",
]
