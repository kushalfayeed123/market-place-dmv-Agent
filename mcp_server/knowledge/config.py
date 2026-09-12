"""Knowledge base configuration — env-driven settings for the KB pipeline.

All settings can be overridden via environment variables. Defaults are safe
for local development; production values are set in deployment env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# Pinned embedding model + dimension. Changing the model changes the index
# namespace (build-new-then-swap reindex); never edit these casually.
EMBEDDING_MODEL_NVIDIA = "nvidia/nv-embedqa-e5-v5"
EMBEDDING_DIM_NVIDIA = 1024


def _int_env(env: dict[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, str(default)))
    except ValueError:
        return default


def _float_env(env: dict[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class KnowledgeConfig:
    """Configuration for the KB ingestion pipeline and query service."""

    # Embedding provider: "nvidia" (production) | "tfidf" (dev/test fallback)
    embedding_provider: str = "nvidia"
    embedding_model: str = EMBEDDING_MODEL_NVIDIA
    embedding_dim: int = EMBEDDING_DIM_NVIDIA
    nvidia_api_key: str = ""
    embedding_batch_size: int = 64
    embedding_timeout: float = 60.0

    # Index identity: namespace suffix carries model+dim so a model change
    # builds a new index instead of corrupting the live one.
    redis_namespace: str = "agent"
    index_version: str = "v1"

    # Sync behaviour
    sync_interval_seconds: int = 3600  # scheduled incremental sync
    backend_base_url: str = "http://localhost:8000"
    backend_api_prefix: str = "/api/v1"

    # Query behaviour
    top_k: int = 5
    min_score: float = 0.30  # below this → "no confident match" → ticket path
    lexical_fallback_weight: float = 0.25  # weight of keyword overlap vs vector score

    # Chunking
    chunk_target_tokens: int = 512
    chunk_overlap_tokens: int = 64

    @property
    def index_namespace(self) -> str:
        """Versioned index namespace: {ns}:kb:v{model-short}:{dim}[#version]."""
        model_short = self.embedding_model.split("/")[-1].replace("-", "_")
        base = f"{self.redis_namespace}:kb:v_{model_short}:{self.embedding_dim}"
        if self.index_version and self.index_version != "v1":
            return f"{base}#{self.index_version}"
        return base


def load_knowledge_config(env: dict[str, str] | None = None) -> KnowledgeConfig:
    """Build a KnowledgeConfig from the environment (or an explicit dict, for tests)."""
    env = env if env is not None else os.environ

    provider = env.get("KB_EMBEDDING_PROVIDER", env.get("EMBEDDING_PROVIDER", "nvidia")).lower()
    model = env.get("KB_EMBEDDING_MODEL", EMBEDDING_MODEL_NVIDIA)
    # Dimension defaults to the pinned dim for the pinned NVIDIA model.
    default_dim = EMBEDDING_DIM_NVIDIA if provider == "nvidia" else 256

    return KnowledgeConfig(
        embedding_provider=provider,
        embedding_model=model,
        embedding_dim=_int_env(env, "KB_EMBEDDING_DIM", default_dim),
        nvidia_api_key=env.get("NVIDIA_API_KEY", ""),
        embedding_batch_size=_int_env(env, "KB_EMBEDDING_BATCH_SIZE", 64),
        embedding_timeout=_float_env(env, "KB_EMBEDDING_TIMEOUT", 60.0),
        redis_namespace=env.get("AGENT_REDIS_NAMESPACE", "agent"),
        index_version=env.get("KB_INDEX_VERSION", "v1"),
        sync_interval_seconds=_int_env(env, "KB_SYNC_INTERVAL_SECONDS", 3600),
        backend_base_url=env.get("BACKEND_API_URL", env.get("BACKEND_BASE_URL", "http://localhost:8000")),
        backend_api_prefix=env.get("BACKEND_API_PREFIX", "/api/v1"),
        top_k=_int_env(env, "KB_TOP_K", 5),
        min_score=_float_env(env, "KB_MIN_SCORE", 0.30),
        lexical_fallback_weight=_float_env(env, "KB_LEXICAL_WEIGHT", 0.25),
        chunk_target_tokens=_int_env(env, "KB_CHUNK_TARGET_TOKENS", 512),
        chunk_overlap_tokens=_int_env(env, "KB_CHUNK_OVERLAP_TOKENS", 64),
    )
