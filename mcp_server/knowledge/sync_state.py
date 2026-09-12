"""Sync state — per-source checkpoints and index version, stored in Redis.

Keys (all under the versioned index namespace):
- ``{ns}:sync:{source}:checksum``   → last synced content checksum of the source doc
- ``{ns}:sync:{source}:ids``        → set of chunk ids currently indexed for that doc
- ``{ns}:sync:last_run``            → ISO timestamp of the last successful sync
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


class SyncState:
    """Redis-backed sync checkpoints (idempotent incremental syncs)."""

    def __init__(self, redis_client: Any, namespace: str):
        self._redis = redis_client
        self._ns = namespace

    def _key(self, *parts: str) -> str:
        return ":".join([self._ns, "sync", *parts])

    async def get_checksum(self, source: str, doc_id: str) -> str | None:
        val = await self._redis.get(self._key(source, doc_id, "checksum"))
        return str(val) if val else None

    async def set_checksum(self, source: str, doc_id: str, checksum: str, ttl: int = 0) -> None:
        key = self._key(source, doc_id, "checksum")
        if ttl > 0:
            await self._redis.set(key, checksum, ex=ttl)
        else:
            await self._redis.set(key, checksum)

    async def get_indexed_ids(self, source: str, doc_id: str) -> set[str]:
        val = await self._redis.smembers(self._key(source, doc_id, "ids"))
        return set(val or [])

    async def set_indexed_ids(self, source: str, doc_id: str, ids: set[str], ttl: int = 0) -> None:
        key = self._key(source, doc_id, "ids")
        if ttl > 0:
            pipe = self._redis.pipeline()
            pipe.delete(key)
            if ids:
                pipe.sadd(key, *ids)
            pipe.expire(key, ttl)
            await pipe.execute()
        else:
            pipe = self._redis.pipeline()
            pipe.delete(key)
            if ids:
                pipe.sadd(key, *ids)
            await pipe.execute()

    async def mark_run(self, stats: dict) -> None:
        await self._redis.set(
            self._key("last_run"),
            json.dumps({"at": datetime.now(timezone.utc).isoformat(), "stats": stats}, default=str),
        )

    async def last_run(self) -> dict | None:
        val = await self._redis.get(self._key("last_run"))
        if not val:
            return None
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None
