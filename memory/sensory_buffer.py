"""Layer 0 — Sensory Buffer (Redis). Raw perception signals with TTL auto-expiry."""

from __future__ import annotations

import json
import time
from uuid import uuid4

import redis.asyncio as aioredis


class SensoryBuffer:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        redis_client: aioredis.Redis | None = None,
    ) -> None:
        if redis_client is not None:
            self._redis = redis_client
        else:
            self._redis = aioredis.from_url(redis_url, decode_responses=True)

    async def aclose(self) -> None:
        await self._redis.aclose()

    async def write_perception(
        self, source: str, data: bytes | str, ttl: int = 60
    ) -> str:
        key = f"perception:{source}:{uuid4()}"
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        payload = json.dumps({
            "source": source,
            "data": data,
            "timestamp": time.time(),
        })
        await self._redis.set(key, payload, ex=ttl)
        return key

    async def read_recent(self, source: str, limit: int = 10) -> list[dict]:
        pattern = f"perception:{source}:*"
        cursor = 0
        keys: list[str] = []
        while True:
            cursor, batch = await self._redis.scan(cursor=cursor, match=pattern)
            keys.extend(batch)
            if cursor == 0:
                break
        if not keys:
            return []
        pipe = self._redis.pipeline()
        for k in keys:
            pipe.get(k)
        values = await pipe.execute()
        perceptions = [
            (k, json.loads(v))
            for k, v in zip(keys, values)
            if v is not None
        ]
        perceptions.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
        return [p[1] for p in perceptions[:limit]]

    async def flush_to_working_memory(
        self,
        key: str,
        summary: str,
        working_memory: object | None = None,
        session_id: str = "default",
    ) -> None:
        raw = await self._redis.get(key)
        if raw is None:
            return
        payload = json.loads(raw)
        await self._redis.delete(key)
        if working_memory is not None:
            from xnch.memory.working_memory import WorkingMemory

            if isinstance(working_memory, WorkingMemory):
                src = payload.get("source", "unknown")
                await working_memory.append_turn(
                    session_id, "perception", f"[{src}] {summary}"
                )
