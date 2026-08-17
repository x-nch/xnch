"""Experience Store — structured experiential reflections derived from decision outcomes.

Mirrors PatternStore: aiosqlite-backed, Bayesian-smoothed confidence,
upsert keyed on context_signature. Lessons are the Summary step's output —
what worked / what failed and why — retrieved during Plan to inform options.

Schema is owned by init_db (see memory/db.py). The store lazily ensures it
once per instance so it also works against pre-migration DBs without running
DDL on every call.
"""
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

_EXPERIENCES_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    experience_id       TEXT PRIMARY KEY,
    context_signature   TEXT NOT NULL UNIQUE,
    intent_class        TEXT NOT NULL,
    action_type         TEXT NOT NULL,
    entity_class        TEXT NOT NULL,
    actor_role          TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    lesson              TEXT NOT NULL,
    insight             TEXT NOT NULL,
    verdict             TEXT NOT NULL,
    applicability       TEXT NOT NULL,
    confidence          REAL NOT NULL,
    observation_count   INTEGER NOT NULL DEFAULT 1,
    created_at          REAL NOT NULL DEFAULT (unixepoch()),
    updated_at          REAL NOT NULL DEFAULT (unixepoch()),
    schema_version      TEXT DEFAULT 'exp-v1'
);

CREATE INDEX IF NOT EXISTS idx_experiences_tuple
    ON experiences(intent_class, entity_class, actor_role);
"""


class ExperienceStore:
    def __init__(self, db_path: Path) -> None:
        self._db = db_path
        self._schema_ensured = False

    async def _ensure_schema(self, db) -> None:
        if self._schema_ensured:
            return
        await db.executescript(_EXPERIENCES_SCHEMA)
        await db.commit()
        self._schema_ensured = True

    async def upsert_experience(
        self,
        context_signature: str,
        intent_class: str,
        action_type: str,
        entity_class: str,
        actor_role: str,
        outcome: str,
        lesson: str,
        insight: str,
        verdict: str,
        applicability: str,
    ) -> None:
        now = time.time()
        async with aiosqlite.connect(self._db) as db:
            await self._ensure_schema(db)
            async with db.execute(
                "SELECT observation_count, lesson, insight, verdict FROM experiences "
                "WHERE context_signature = ?",
                (context_signature,),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                count = row[0] + 1
                confidence = self._bayes_confidence(count)
                await db.execute(
                    """UPDATE experiences SET outcome=?, lesson=?, insight=?, verdict=?,
                       applicability=?, confidence=?, observation_count=?, updated_at=?
                       WHERE context_signature=?""",
                    (outcome, lesson, insight, verdict, applicability,
                     confidence, count, now, context_signature),
                )
            else:
                confidence = self._bayes_confidence(1)
                await db.execute(
                    """INSERT INTO experiences
                       (experience_id, context_signature, intent_class, action_type, entity_class,
                        actor_role, outcome, lesson, insight, verdict, applicability,
                        confidence, observation_count, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (str(uuid4()), context_signature, intent_class, action_type, entity_class,
                     actor_role, outcome, lesson, insight, verdict, applicability,
                     confidence, 1, now, now),
                )
            await db.commit()

    def _bayes_confidence(self, count: int) -> float:
        """Beta(α=1, β=1) posterior mean over repeated observations of a lesson."""
        return round((count + 1) / (count + 2), 4)

    async def fetch_for_manifest(
        self,
        intent_class: str,
        entity_class: str,
        actor_role: str,
        max_experiences: int = 10,
    ) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db) as db:
            await self._ensure_schema(db)
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT experience_id, context_signature, intent_class, action_type,
                          entity_class, actor_role, outcome, lesson, insight, verdict,
                          applicability, confidence, observation_count, created_at
                   FROM experiences
                   WHERE intent_class = ? AND entity_class = ? AND actor_role = ?
                   ORDER BY confidence DESC LIMIT ?""",
                (intent_class, entity_class, actor_role, max_experiences),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]
