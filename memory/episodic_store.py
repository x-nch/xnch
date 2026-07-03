"""Episodic Store — records every decision outcome for the learning loop."""
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite


class EpisodicStore:
    def __init__(self, db_path: Path) -> None:
        self._db = db_path

    async def create_episode(
        self,
        decision_id: str,
        intent_class: str,
        action_type: str,
        entity_class: str,
        actor_role: str,
        context_snapshot: dict[str, Any],
        generation_path: str = "MODEL",
    ) -> str:
        episode_id = str(uuid4())
        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                """INSERT INTO episodes
                   (episode_id, decision_id, intent_class, action_type, entity_class,
                    actor_role, context_snapshot, generation_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (episode_id, decision_id, intent_class, action_type, entity_class,
                 actor_role, json.dumps(context_snapshot), generation_path, time.time()),
            )
            await db.commit()
        return episode_id

    async def complete_episode(
        self,
        decision_id: str,
        outcome: str,
        observed_state_delta: dict[str, Any],
        side_effects: list[str],
        duration_ms: int,
        anomalies: list[str],
    ) -> str | None:
        snapshot = {
            "observed_state_delta": observed_state_delta,
            "side_effects_observed": side_effects,
            "duration_ms": duration_ms,
            "anomalies": anomalies,
        }
        async with aiosqlite.connect(self._db) as db:
            async with db.execute(
                "SELECT episode_id FROM episodes WHERE decision_id = ?", (decision_id,)
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return None
            episode_id = row[0]
            await db.execute(
                """UPDATE episodes SET outcome = ?, context_snapshot = json_patch(context_snapshot, ?),
                   completed_at = ? WHERE episode_id = ?""",
                (outcome, json.dumps(snapshot), time.time(), episode_id),
            )
            await db.commit()
        return episode_id

    async def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ) as cursor:
                row = await cursor.fetchone()
        return dict(row) if row else None

    async def write_prediction_update(
        self,
        episode_id: str,
        prediction_delta: float,
        early_reextraction_flag: bool,
    ) -> None:
        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                """UPDATE episodes SET prediction_delta = ?, early_reextraction_flag = ?
                   WHERE episode_id = ?""",
                (prediction_delta, int(early_reextraction_flag), episode_id),
            )
            await db.commit()

    async def fetch_for_extraction(
        self,
        intent_class: str,
        action_type: str,
        entity_class: str,
        actor_role: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch completed, non-rule-based episodes for a given context tuple."""
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT episode_id, outcome, prediction_delta
                   FROM episodes
                   WHERE intent_class = ? AND action_type = ? AND entity_class = ?
                     AND actor_role = ? AND outcome IS NOT NULL
                     AND generation_path = 'MODEL'
                   ORDER BY created_at DESC LIMIT ?""",
                (intent_class, action_type, entity_class, actor_role, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def fetch_for_manifest(
        self,
        intent_class: str,
        entity_class: str,
        actor_role: str,
        lookback_days: int = 30,
        max_episodes: int = 20,
    ) -> list[dict[str, Any]]:
        cutoff = time.time() - lookback_days * 86400
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT episode_id, action_type, entity_class, outcome, created_at, completed_at
                   FROM episodes
                   WHERE intent_class = ? AND entity_class = ? AND actor_role = ?
                     AND created_at >= ? AND outcome IS NOT NULL
                   ORDER BY created_at DESC LIMIT ?""",
                (intent_class, entity_class, actor_role, cutoff, max_episodes),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_distinct_tuples(self) -> list[tuple[str, str, str, str]]:
        """All (intent_class, action_type, entity_class, actor_role) tuples with completed episodes."""
        async with aiosqlite.connect(self._db) as db:
            async with db.execute(
                """SELECT DISTINCT intent_class, action_type, entity_class, actor_role
                   FROM episodes WHERE outcome IS NOT NULL"""
            ) as cursor:
                rows = await cursor.fetchall()
        return [tuple(r) for r in rows]

    async def get_flagged_for_early_extraction(self) -> list[tuple[str, str, str, str]]:
        async with aiosqlite.connect(self._db) as db:
            async with db.execute(
                """SELECT DISTINCT intent_class, action_type, entity_class, actor_role
                   FROM episodes WHERE early_reextraction_flag = 1 AND outcome IS NOT NULL"""
            ) as cursor:
                rows = await cursor.fetchall()
        return [tuple(r) for r in rows]

    async def clear_early_extraction_flags(self) -> None:
        async with aiosqlite.connect(self._db) as db:
            await db.execute("UPDATE episodes SET early_reextraction_flag = 0")
            await db.commit()
