"""Sync orchestration — full and incremental knowledge-base syncs.

Usage (from repo root):
    python -m Agent.mcp_server.knowledge.sync                # incremental
    python -m Agent.mcp_server.knowledge.sync --full         # re-chunk everything
    python -m Agent.mcp_server.knowledge.sync --query "..."  # test a query after sync
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from .config import KnowledgeConfig, load_knowledge_config
from .embedders import create_embedder
from .indexer import KBIndexer, SyncStats
from .service import KnowledgeService
from .sync_state import SyncState


def _default_collectors(config: KnowledgeConfig) -> list:
    from .collectors import CatalogCollector, DocumentsCollector

    return [
        DocumentsCollector(config.backend_base_url, config.backend_api_prefix),
        CatalogCollector(config.backend_base_url, config.backend_api_prefix),
    ]


async def run_sync(
    redis_client: Any,
    config: KnowledgeConfig | None = None,
    *,
    full: bool = False,
    collectors: list | None = None,
) -> SyncStats:
    """Run a knowledge sync (incremental by default, full when ``full=True``).

    Idempotent: unchanged documents are skipped via content checksums;
    changed/archived documents are re-chunked and their stale chunks deleted.
    """
    config = config or load_knowledge_config()
    embedder = create_embedder(
        config.embedding_provider,
        api_key=config.nvidia_api_key,
        model=config.embedding_model,
        dim=config.embedding_dim,
        batch_size=config.embedding_batch_size,
        timeout=config.embedding_timeout,
    )
    indexer = KBIndexer(redis_client, embedder, config)
    total = SyncStats()

    for collector in (collectors if collectors is not None else _default_collectors(config)):
        try:
            async for doc in collector.collect():
                stats = await indexer.sync_document(doc, incremental=not full)
                total.sources_seen += stats.sources_seen
                total.docs_changed += stats.docs_changed
                total.docs_unchanged += stats.docs_unchanged
                total.chunks_upserted += stats.chunks_upserted
                total.chunks_deleted += stats.chunks_deleted
                total.errors += stats.errors
                total.details.extend(stats.details)
        except Exception as exc:  # noqa: BLE001 — collector crash must not abort the sync
            total.errors += 1
            total.details.append({"collector": getattr(collector, "name", "?"), "error": str(exc)})

    await SyncState(redis_client, config.index_namespace).mark_run(total.to_dict())
    return total


async def _main(full: bool, test_query: str | None) -> int:
    import redis.asyncio as redis

    from dotenv import load_dotenv

    load_dotenv()
    config = load_knowledge_config()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # Normalize Upstash-style https:// URLs to rediss:// and build the client
    # exactly like the gateway does (gateway/main.py `_create_redis`): disable
    # TLS verification for rediss:// so a rotated/expired Upstash server cert
    # doesn't block the standalone sync CLI with CERTIFICATE_VERIFY_FAILED.
    if redis_url.startswith("https://"):
        redis_url = "rediss://" + redis_url[len("https://"):]
    elif redis_url.startswith("http://"):
        redis_url = "redis://" + redis_url[len("http://"):]
    elif not redis_url.startswith(("redis://", "rediss://", "unix://")):
        redis_url = "redis://" + redis_url
    redis_kwargs: dict[str, object] = {"decode_responses": True}
    if redis_url.startswith("rediss://"):
        redis_kwargs["ssl_cert_reqs"] = None
    client = redis.from_url(redis_url, **redis_kwargs)
    try:
        await client.ping()
    except Exception as exc:  # noqa: BLE001
        print(f"[kb-sync] ERROR: cannot reach Redis: {exc}")
        return 2

    mode = "full" if full else "incremental"
    print(f"[kb-sync] starting {mode} sync (index: {config.index_namespace})")
    stats = await run_sync(client, config, full=full)
    print(f"[kb-sync] done: {stats.to_dict()}")
    for detail in stats.details[:10]:
        print(f"[kb-sync]   error: {detail}")

    if test_query:
        embedder = create_embedder(
            config.embedding_provider,
            api_key=config.nvidia_api_key,
            model=config.embedding_model,
            dim=config.embedding_dim,
        )
        service = KnowledgeService(client, embedder, config)
        answer = await service.search(test_query)
        print(f"[kb-sync] query {test_query!r} → status={answer.status} top_score={answer.top_score}")
        for r in answer.results[:5]:
            print(f"  [{r.score:.4f}] {r.doc_id}: {r.title} — {r.text[:100]}")

    await client.aclose()
    return 0 if stats.errors == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Knowledge-base sync (incremental by default).")
    parser.add_argument("--full", action="store_true", help="Re-chunk and re-embed every document.")
    parser.add_argument("--query", default=None, help="Run a test search after syncing.")
    args = parser.parse_args(argv)
    return asyncio.run(_main(full=args.full, test_query=args.query))


if __name__ == "__main__":
    sys.exit(main())
