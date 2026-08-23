"""Deep memory-tier health probes — real round-trips, not just "process up".

Each tier gets a probe that exercises the actual failure modes this stack
has hit before:

- redis: SET-with-EX canary must come back with a live TTL, and any value
  already sitting in the canary slot MUST have an expiry set (catches both
  a server ignoring EX and code writing immortal keys into an ephemeral tier).
- postgres: latency of a real query against the episodic `episodes` table.
- kuzu: full write->read round-trip of a probe node (write path previously
  broke silently while mocked tests stayed green — this makes that visible).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .metrics import MEMORY_TIER_LAST_SUCCESS, MEMORY_TIER_PROBE_SECONDS, MEMORY_TIER_UP

logger = logging.getLogger(__name__)

_REDIS_CANARY_PREFIX = "obs:canary:"
_REDIS_SENTINEL_KEY = "obs:canary-sentinel"
_KUZU_PROBE_ID = "obs-probe-node"
_KUZU_PROBE_NAME = "_obs_probe"


@dataclass
class ProbeResult:
    tier: str
    ok: bool
    latency_ms: float = 0.0
    detail: str = ""


async def check_redis_ttl_canary(redis_client: Any) -> ProbeResult:
    start = time.perf_counter()
    key = f"{_REDIS_CANARY_PREFIX}{uuid4().hex}"
    try:
        await redis_client.set(key, "1", ex=2)
        ttl = int(await redis_client.ttl(key))
        if ttl <= 0 or ttl > 2:
            return ProbeResult(
                tier="redis",
                ok=False,
                latency_ms=_ms(start),
                detail=f"canary TTL out of range after SET EX 2 (ttl={ttl})",
            )
        sentinel_ttl = int(await redis_client.ttl(_REDIS_SENTINEL_KEY))
        if await redis_client.exists(_REDIS_SENTINEL_KEY) and sentinel_ttl <= 0:
            return ProbeResult(
                tier="redis",
                ok=False,
                latency_ms=_ms(start),
                detail=f"sentinel key {_REDIS_SENTINEL_KEY!r} exists without expiry",
            )
        await redis_client.delete(key)
        return ProbeResult(
            tier="redis",
            ok=True,
            latency_ms=_ms(start),
            detail=f"ttl={ttl}",
        )
    except Exception as exc:
        return ProbeResult(tier="redis", ok=False, latency_ms=_ms(start), detail=str(exc))


_EPISODIC_PROBE_SQL = "SELECT count(*) FROM episodes WHERE archived = FALSE"


async def check_postgres_episodic(pool: Any) -> ProbeResult:
    start = time.perf_counter()
    try:
        async with pool.acquire() as conn:
            count = await conn.fetchval(_EPISODIC_PROBE_SQL)
        return ProbeResult(
            tier="postgres",
            ok=True,
            latency_ms=_ms(start),
            detail=f"episodes(unarchived)={count}",
        )
    except Exception as exc:
        return ProbeResult(tier="postgres", ok=False, latency_ms=_ms(start), detail=str(exc))


async def check_kuzu_roundtrip(graph_store: Any) -> ProbeResult:
    """Write then read back a fixed probe entity; idempotent upsert, no growth."""
    start = time.perf_counter()
    try:
        graph_store.upsert_entity(
            id=_KUZU_PROBE_ID, name=_KUZU_PROBE_NAME, type_="probe"
        )
        fetched = graph_store.get_entity_by_name(_KUZU_PROBE_NAME)
        if not fetched:
            return ProbeResult(
                tier="kuzu",
                ok=False,
                latency_ms=_ms(start),
                detail="probe entity write succeeded but read-back returned None",
            )
        return ProbeResult(
            tier="kuzu",
            ok=True,
            latency_ms=_ms(start),
            detail=f"probe_id={fetched.get('entity_id', _KUZU_PROBE_ID)}",
        )
    except Exception as exc:
        return ProbeResult(tier="kuzu", ok=False, latency_ms=_ms(start), detail=str(exc))


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


@dataclass
class DeepHealthRunner:
    """Runs all configured probes on an interval; publishes gauges + JSON summary."""

    redis_client: Any | None = None
    pg_pool: Any | None = None
    graph_store: Any | None = None
    interval_s: float = 30.0
    last_results: dict[str, ProbeResult] = field(default=None, init=False, repr=False)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)

    def probe_map(self) -> dict[str, Any]:
        probes: dict[str, Any] = {}
        if self.redis_client is not None:
            probes["redis"] = lambda: check_redis_ttl_canary(self.redis_client)
        if self.pg_pool is not None:
            probes["postgres"] = lambda: check_postgres_episodic(self.pg_pool)
        if self.graph_store is not None:
            probes["kuzu"] = lambda: check_kuzu_roundtrip(self.graph_store)
        return probes

    async def run_once(self) -> dict[str, ProbeResult]:
        results: dict[str, ProbeResult] = {}
        for tier, probe in self.probe_map().items():
            result = await probe()
            results[tier] = result
            self.last_results = {**(self.last_results or {}), tier: result}
            MEMORY_TIER_UP.labels(tier=tier).set(1.0 if result.ok else 0.0)
            MEMORY_TIER_PROBE_SECONDS.labels(tier=tier).set(result.latency_ms / 1000.0)
            if result.ok:
                MEMORY_TIER_LAST_SUCCESS.labels(tier=tier).set(time.time())
            else:
                logger.warning("deep health probe failed (%s): %s", tier, result.detail)
        return results

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("deep health loop error: %s", exc)
            await asyncio.sleep(self.interval_s)

    @staticmethod
    def to_summary(results: dict[str, ProbeResult]) -> dict[str, Any]:
        return {
            tier: {
                "ok": r.ok,
                "latency_ms": r.latency_ms,
                "detail": r.detail,
            }
            for tier, r in results.items()
        }
