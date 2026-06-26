"""Layer 1 — Working Memory (Redis). Active session context with TTL."""

from __future__ import annotations

import json
import time
from typing import Any

import redis.asyncio as aioredis


class WorkingMemory:
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

    async def set_context(
        self, session_id: str, key: str, value: Any, ttl: int = 3600
    ) -> None:
        hkey = f"session:{session_id}:{key}"
        await self._redis.hset(hkey, mapping={"value": json.dumps(value)})
        await self._redis.expire(hkey, ttl)

    async def get_context(self, session_id: str, key: str) -> Any | None:
        hkey = f"session:{session_id}:{key}"
        raw = await self._redis.hget(hkey, "value")
        if raw is None:
            return None
        return json.loads(raw)

    async def get_full_session(self, session_id: str) -> dict[str, Any]:
        pattern = f"session:{session_id}:*"
        cursor = 0
        keys: list[str] = []
        while True:
            cursor, batch = await self._redis.scan(cursor=cursor, match=pattern)
            keys.extend(batch)
            if cursor == 0:
                break
        result: dict[str, Any] = {}
        hash_keys = [k for k in keys if k.split(":", 2)[2] != "turns"]
        if not hash_keys:
            return result
        pipe = self._redis.pipeline()
        for k in hash_keys:
            pipe.hgetall(k)
        values = await pipe.execute()
        for k, v in zip(hash_keys, values):
            if not v:
                continue
            key_name = k.split(":", 2)[2]
            result[key_name] = json.loads(v.get("value", "null"))
        return result

    async def clear_session(self, session_id: str) -> None:
        pattern = f"session:{session_id}:*"
        cursor = 0
        keys: list[str] = []
        while True:
            cursor, batch = await self._redis.scan(cursor=cursor, match=pattern)
            keys.extend(batch)
            if cursor == 0:
                break
        if keys:
            await self._redis.delete(*keys)

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        turn_key = f"session:{session_id}:turns"
        turn = json.dumps({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        await self._redis.rpush(turn_key, turn)

    async def get_turns(
        self, session_id: str, last_n: int = 20
    ) -> list[dict[str, Any]]:
        turn_key = f"session:{session_id}:turns"
        length = await self._redis.llen(turn_key)
        start = max(0, length - last_n)
        raw = await self._redis.lrange(turn_key, start, -1)
        return [json.loads(t) for t in raw]
