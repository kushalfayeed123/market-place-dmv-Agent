"""Ingestion script: index marketplace knowledge into Redis.

Usage (from repo root, Redis running):
    python -m Agent.mcp_server.ingest_vectors
    python -m Agent.mcp_server.ingest_vectors --clear
    python -m Agent.mcp_server.ingest_vectors --query "warm winter jacket"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request

SEED_DOCS: list[dict] = [
    {
        "id": "policy-returns",
        "text": (
            "Return policy: Buyers may return most unused items within 30 days of "
            "delivery for a full refund. The seller covers return shipping for "
            "defective or mis-described items."
        ),
        "metadata": {"kind": "policy", "topic": "returns"},
    },
    {
        "id": "policy-shipping",
        "text": (
            "Shipping policy: Standard delivery takes 3-5 business days. Express "
            "delivery takes 1-2 business days. Free standard shipping on orders "
            "over $50. Tracking is available for all shipments."
        ),
        "metadata": {"kind": "policy", "topic": "shipping"},
    },
    {
        "id": "faq-payment-methods",
        "text": (
            "We accept major credit cards, debit cards, and digital wallets. "
            "Payments are processed securely. High-value orders may require "
            "additional confirmation."
        ),
        "metadata": {"kind": "faq", "topic": "payments"},
    },
    {
        "id": "faq-track-order",
        "text": (
            "To track your order, open your orders list and select the order to "
            "see fulfillment status and tracking number. Email updates are sent "
            "at each shipment milestone."
        ),
        "metadata": {"kind": "faq", "topic": "orders"},
    },
    {
        "id": "guide-seller-onboarding",
        "text": (
            "Merchant onboarding: create a merchant profile, complete KYC "
            "verification, list products with clear titles and descriptions, set "
            "inventory per variant, and connect a payout account to receive funds."
        ),
        "metadata": {"kind": "guide", "topic": "merchants"},
    },
]


def _fetch_backend_docs(base_url: str, prefix: str, timeout: int = 15) -> list[dict]:
    """Pull products + categories from the backend API (public endpoints)."""
    docs: list[dict] = []

    def _get(path: str):
        try:
            with urllib.request.urlopen(base_url + prefix + path, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            print(f"[ingest] backend {path} unavailable: {exc}")
            return None

    categories = _get("/catalog/categories") or []
    cat_names = {c.get("id"): c.get("name", "") for c in categories if isinstance(c, dict)}

    products = _get("/catalog/products?limit=100") or []
    items = products.get("items", products) if isinstance(products, dict) else products
    for p in items if isinstance(items, list) else []:
        if not isinstance(p, dict):
            continue
        pid = p.get("id", p.get("product_id", "unknown"))
        cat = cat_names.get(p.get("category_id"), p.get("category", ""))
        text = " ".join(
            str(x)
            for x in [p.get("name", ""), p.get("description", ""), cat]
            if x
        ).strip()
        if text:
            docs.append(
                {"id": f"product-{pid}", "text": text, "metadata": {"kind": "product"}}
            )
    return docs

async def _run(clear: bool, query: str | None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    import redis.asyncio as redis
    from Agent.mcp_server.vector_store import TfidfEmbedder, VectorStore

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    namespace = os.getenv("AGENT_REDIS_NAMESPACE", "agent")
    backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000")
    backend_prefix = os.getenv("BACKEND_API_PREFIX", "/api/v1")

    client = redis.from_url(redis_url, decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:
        print(f"[ingest] ERROR: cannot reach Redis at {redis_url}: {exc}")
        print("[ingest] Start Redis first (e.g. `redis-server`) or set REDIS_URL.")
        return 2

    store = VectorStore(redis_client=client, embedder=TfidfEmbedder(), namespace=namespace)
    if clear:
        removed = await store.clear()
        print(f"[ingest] cleared {removed} existing documents")

    backend_docs = await asyncio.to_thread(_fetch_backend_docs, backend_url, backend_prefix)
    docs = list(SEED_DOCS) + backend_docs
    print(f"[ingest] indexing {len(docs)} docs")

    embedder = TfidfEmbedder()
    embedder.fit([d["text"] for d in docs])
    store = VectorStore(redis_client=client, embedder=embedder, namespace=namespace)
    count = await store.upsert_many(docs)
    print(f"[ingest] indexed {count} documents into namespace {namespace!r}")

    if query:
        print(f"[ingest] search: {query!r}")
        for r in await store.search(query, top_k=5):
            print(f"  [{r['score']:.4f}] {r['id']}: {r['text'][:120]}")

    await client.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index marketplace knowledge into Redis.")
    parser.add_argument("--clear", action="store_true", help="Wipe the vector index first.")
    parser.add_argument("--query", default=None, help="Run a test search after indexing.")
    args = parser.parse_args(argv)
    return asyncio.run(_run(clear=args.clear, query=args.query))


if __name__ == "__main__":
    sys.exit(main())

