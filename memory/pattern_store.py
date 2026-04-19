"""Pattern Store — aggregated patterns derived from episodic history."""
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite


class PatternStore:
    def __init__(self, db_path: Path) -> None:
        self._db = db_path

    async def upsert_pattern(
        self,
        context_signature: str,
        intent_class: str,
        action_type: str,
        entity_class: str,
        actor_role: str,
        success_rate: float,
        confidence: float,
        observation_count: int,
        avg_prediction_delta: float | None,
        extraction_run_id: str,
    ) -> None:
        now = time.time()
        async with aiosqlite.connect(self._db) as db:
            async with db.execute(
                "SELECT pattern_id FROM patterns WHERE context_signature = ?",
                (context_signature,),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                await db.execute(
                    """UPDATE patterns SET success_rate=?, confidence=?, observation_count=?,
                       avg_prediction_delta=?, extraction_run_id=?, updated_at=?
                       WHERE context_signature=?""",
                    (success_rate, confidence, observation_count, avg_prediction_delta,
                     extraction_run_id, now, context_signature),
                )
            else:
                await db.execute(
                    """INSERT INTO patterns
                       (pattern_id, context_signature, intent_class, action_type, entity_class,
                        actor_role, success_rate, confidence, observation_count,
                        avg_prediction_delta, extraction_run_id, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (str(uuid4()), context_signature, intent_class, action_type, entity_class,
                     actor_role, success_rate, confidence, observation_count,
                     avg_prediction_delta, extraction_run_id, now, now),
                )
            await db.commit()

    async def fetch_for_manifest(
        self,
        intent_class: str,
        entity_class: str,
        actor_role: str,
        max_patterns: int = 10,
    ) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT pattern_id, context_signature, success_rate, confidence, observation_count
                   FROM patterns
                   WHERE intent_class = ? AND entity_class = ? AND actor_role = ?
                   ORDER BY confidence DESC LIMIT ?""",
                (intent_class, entity_class, actor_role, max_patterns),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def fetch_low_success(
        self,
        max_success_rate: float = 0.4,
        min_confidence: float = 0.6,
    ) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM patterns
                   WHERE success_rate < ? AND confidence > ?
                   ORDER BY confidence DESC""",
                (max_success_rate, min_confidence),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def fetch_all(self) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM patterns") as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]
