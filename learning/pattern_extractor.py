"""Pattern Extractor — 6h APScheduler job.

Queries PgEpisodicStore decision_episodes since last_run.
Falls back to EpisodicStore (SQLite) when PgEpisodicStore not available.
Groups by (intent_class, action_type, entity_class, actor_role).
Upserts into PatternStore with Bayesian-smoothed confidence.
Last-run timestamp tracked in SQLite extraction_tracker table.
"""
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import aiosqlite

from ..memory.pattern_store import PatternStore
from ..config import settings

if TYPE_CHECKING:
    from ..memory.pg_episodic_store import PgEpisodicStore

logger = logging.getLogger(__name__)


def _context_signature(intent_class: str, action_type: str, entity_class: str, actor_role: str) -> str:
    canonical = "|".join([
        intent_class.lower(), action_type.lower(), entity_class.lower(), actor_role.lower()
    ])
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


class PatternExtractor:
    def __init__(
        self,
        store,
        patterns: PatternStore,
        db_path: Path | None = None,
    ) -> None:
        self._store = store
        self._patterns = patterns
        self._db = db_path or settings.db_path
        self._is_pg = type(store).__name__ == "PgEpisodicStore"

    async def run(self) -> int:
        last_run = await self._get_last_run()

        if self._is_pg:
            return await self._run_pg(last_run)
        return await self._run_sqlite(last_run)

    async def _run_pg(self, last_run: float | None) -> int:
        since = last_run if last_run else time.time() - 86400 * 30
        since_dt = datetime.fromtimestamp(since, tz=timezone.utc)
        episodes = await self._store.fetch_decision_episodes_since(since_dt)

        if not episodes:
            await self._set_last_run(time.time())
            return 0

        groups: dict[tuple[str, str, str, str], list[dict]] = {}
        for ep in episodes:
            key = (ep["intent_class"], ep["action_type"], ep["entity_class"], ep["actor_role"])
            groups.setdefault(key, []).append(ep)

        written = 0
        run_id = str(uuid4())

        for (intent_class, action_type, entity_class, actor_role), batch in groups.items():
            n, sr, conf, ad = self._compute_pattern(batch)
            if n < settings.pattern_min_observations:
                continue
            await self._upsert(intent_class, action_type, entity_class, actor_role, sr, conf, n, ad, run_id)
            written += 1

        await self._set_last_run(time.time())
        return written

    async def _run_sqlite(self, last_run: float | None) -> int:
        from ..memory.episodic_store import EpisodicStore
        store: EpisodicStore = self._store
        tuples = await store.get_distinct_tuples()
        run_id = str(uuid4())
        written = 0

        for intent_class, action_type, entity_class, actor_role in tuples:
            episodes = await store.fetch_for_extraction(
                intent_class, action_type, entity_class, actor_role
            )
            n, sr, conf, ad = self._compute_pattern(episodes)
            if n < settings.pattern_min_observations:
                continue
            await self._upsert(intent_class, action_type, entity_class, actor_role, sr, conf, n, ad, run_id)
            written += 1

        await self._set_last_run(time.time())
        return written

    def _compute_pattern(
        self,
        episodes: list[dict],
    ) -> tuple[int, float, float, float | None]:
        n = len(episodes)
        successes = sum(1 for ep in episodes if ep.get("outcome") == "SUCCESS")
        success_rate = successes / n if n else 0.0
        confidence = (successes + 1) / (n + 2)
        deltas = [ep["prediction_delta"] for ep in episodes if ep.get("prediction_delta") is not None]
        avg_delta = sum(deltas) / len(deltas) if deltas else None
        return n, success_rate, confidence, avg_delta

    async def _upsert(
        self,
        intent_class: str,
        action_type: str,
        entity_class: str,
        actor_role: str,
        success_rate: float,
        confidence: float,
        observation_count: int,
        avg_prediction_delta: float | None,
        run_id: str,
    ) -> None:
        sig = _context_signature(intent_class, action_type, entity_class, actor_role)
        await self._patterns.upsert_pattern(
            context_signature=sig,
            intent_class=intent_class,
            action_type=action_type,
            entity_class=entity_class,
            actor_role=actor_role,
            success_rate=round(success_rate, 4),
            confidence=round(confidence, 4),
            observation_count=observation_count,
            avg_prediction_delta=round(avg_prediction_delta, 4) if avg_prediction_delta is not None else None,
            extraction_run_id=run_id,
        )
        logger.info(
            "Pattern written: %s|%s|%s|%s  success_rate=%.2f  confidence=%.2f  n=%d",
            intent_class, action_type, entity_class, actor_role,
            success_rate, confidence, observation_count,
        )

    async def _get_last_run(self) -> float | None:
        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS extraction_tracker (key TEXT PRIMARY KEY, value TEXT)"
            )
            async with db.execute(
                "SELECT value FROM extraction_tracker WHERE key = 'last_run'"
            ) as cursor:
                row = await cursor.fetchone()
        return float(row[0]) if row else None

    async def _set_last_run(self, ts: float) -> None:
        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                "INSERT OR REPLACE INTO extraction_tracker (key, value) VALUES ('last_run', ?)",
                (str(ts),),
            )
            await db.commit()