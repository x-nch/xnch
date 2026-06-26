"""KV Cache — Redis-backed session deduplication and rate limiting."""
import json
import math
import time
from typing import Any

import redis.asyncio as aioredis

from ..config import settings
from xnch.security.trust_model import TrustLevel, get_trust_level


_RATE_LIMITS_BY_TRUST: dict[TrustLevel, int] = {
    TrustLevel.SYSTEM: 999999,
    TrustLevel.OWNER: 1000,
    TrustLevel.TRUSTED_AGENT: 100,
    TrustLevel.EXTERNAL_AGENT: 10,
    TrustLevel.UNTRUSTED: 0,
}


class KVCache:
    def __init__(self, redis_url: str) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)

    @property
    def redis_client(self) -> aioredis.Redis:
        return self._redis

    async def aclose(self) -> None:
        await self._redis.aclose()

    # ------------------------------------------------------------------
    # Session deduplication (Step 2a)
    # ------------------------------------------------------------------

    async def get_session(self, idempotency_key: str) -> dict[str, Any] | None:
        raw = await self._redis.get(f"session:{idempotency_key}")
        return json.loads(raw) if raw else None

    async def set_session(
        self, idempotency_key: str, session_context: dict[str, Any], ttl_s: int | None = None
    ) -> None:
        ttl = ttl_s or settings.session_ttl_s
        await self._redis.set(
            f"session:{idempotency_key}", json.dumps(session_context), ex=ttl
        )

    async def reset_session_ttl(self, idempotency_key: str) -> None:
        await self._redis.expire(f"session:{idempotency_key}", settings.session_ttl_s)

    async def delete_session(self, idempotency_key: str) -> None:
        await self._redis.delete(f"session:{idempotency_key}")

    # ------------------------------------------------------------------
    # Rate limiting (Step 2a — INCR per actor per minute bucket)
    # ------------------------------------------------------------------

    async def check_rate_limit(self, actor_id: str) -> bool:
        """Return True if request is within limit, False if exceeded."""
        bucket = math.floor(time.time() / 60)
        key = f"rate:{actor_id}:{bucket}"
        count = await self._redis.incr(key)
        await self._redis.expire(key, 60)
        return count <= settings.rate_limit_per_minute

    # ------------------------------------------------------------------
    # Rate limiting per actor (actor role scoped)
    # ------------------------------------------------------------------

    async def check_actor_rate(self, actor_role: str, endpoint: str, limit: int | None = None, window_seconds: int = 60) -> bool:
        trust_level = get_trust_level(actor_role)
        if limit is None:
            limit = _RATE_LIMITS_BY_TRUST.get(trust_level, 0)
        if limit <= 0:
            return False
        bucket = math.floor(time.time() / window_seconds)
        key = f"ratelimit:{actor_role}:{endpoint}:{bucket}"
        count = await self._redis.incr(key)
        await self._redis.expire(key, window_seconds)
        return count <= limit

    # ------------------------------------------------------------------
    # Ping
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        try:
            return await self._redis.ping()
        except Exception:
            return False
