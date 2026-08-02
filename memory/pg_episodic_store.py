"""PostgreSQL + pgvector episodic store — Layer 2 memory.

Single store serving BOTH query modes of the four-tier design:
- exact-match: SQL WHERE on intent/entity/actor tuple + recency (replaces
  the ad hoc SQLite EpisodicStore)
- semantic: pgvector cosine distance (<=>) over MiniLM-L6-v2 embeddings
  (replaces agentmemory/ChromaDB retrieve_similar)

Owns the PG schema (episodes, decision_episodes, patterns) and applies it
idempotently on connect.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

import asyncpg

from xnch.config import settings
from xnch.memory.embeddings import embed_text

CATEGORY = "episodes"


_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS episodes (
    id            UUID PRIMARY KEY,
    type          TEXT NOT NULL DEFAULT 'episode',
    raw_text      TEXT NOT NULL DEFAULT '',
    summary       TEXT NOT NULL DEFAULT '',
    embedding     vector(384),
    importance    FLOAT DEFAULT 1.0,
    recall_count  INT DEFAULT 0,
    last_recalled TIMESTAMPTZ,
    decay_score   FLOAT DEFAULT 1.0,
    archived      BOOLEAN DEFAULT FALSE,
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_episodes_archived_ts ON episodes(archived, timestamp DESC);

CREATE TABLE IF NOT EXISTS decision_episodes (
    episode_id              UUID PRIMARY KEY,
    decision_id             TEXT NOT NULL,
    intent_class            TEXT NOT NULL,
    action_type             TEXT NOT NULL,
    entity_class            TEXT NOT NULL,
    actor_role              TEXT NOT NULL,
    outcome                 TEXT,
    prediction_delta        FLOAT,
    early_reextraction_flag BOOLEAN DEFAULT FALSE,
    context_snapshot        JSONB,
    scores_json             JSONB,
    generation_path         TEXT DEFAULT 'MODEL',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at            TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_decision_episodes_tuple
    ON decision_episodes(intent_class, action_type, entity_class, actor_role);
CREATE INDEX IF NOT EXISTS idx_decision_episodes_decision
    ON decision_episodes(decision_id);
CREATE INDEX IF NOT EXISTS idx_decision_episodes_created
    ON decision_episodes(created_at DESC);

CREATE TABLE IF NOT EXISTS patterns (
    pattern_id           UUID PRIMARY KEY,
    context_signature    TEXT NOT NULL UNIQUE,
    intent_class         TEXT NOT NULL,
    action_type          TEXT NOT NULL,
    entity_class         TEXT NOT NULL,
    actor_role           TEXT NOT NULL,
    success_rate         FLOAT NOT NULL,
    confidence           FLOAT NOT NULL,
    observation_count    INT NOT NULL,
    avg_prediction_delta FLOAT,
    extraction_run_id    TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class PgEpisodicStore:
    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or settings.postgres_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=1, max_size=5, command_timeout=60
        )
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------ #
    # Episodes (semantic + recent)
    # ------------------------------------------------------------------ #

    async def store_episode(
        self,
        type_: str,
        raw_text: str | None = None,
        summary: str | None = None,
        embedding: list[float] | None = None,
        importance: float = 1.0,
    ) -> str:
        memory_id = str(uuid.uuid4())
        text = raw_text or summary or ""
        if embedding is None and text:
            embedding = embed_text(text[:512])
        if not self._pool:
            return memory_id
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO episodes (id, type, raw_text, summary, embedding, importance)
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                memory_id, type_, raw_text or "", summary or "",
                _to_vector(embedding), importance,
            )
        return memory_id

    async def retrieve_similar(
        self,
        embedding: list[float] | None = None,
        query_text: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        if embedding is None and query_text:
            embedding = embed_text(query_text[:512])

        async with self._pool.acquire() as conn:
            if embedding is not None:
                rows = await conn.fetch(
                    """SELECT id, type, raw_text, summary, importance, recall_count,
                              last_recalled, timestamp, decay_score, archived,
                              1 - (embedding <=> $1) AS similarity
                       FROM episodes
                       WHERE archived = FALSE AND embedding IS NOT NULL
                       ORDER BY embedding <=> $1
                       LIMIT $2""",
                    _to_vector(embedding), top_k,
                )
                processed = []
                for r in rows:
                    sim = float(r["similarity"])
                    if sim < min_score:
                        continue
                    processed.append(_episode_row(r, sim=sim))
                return processed

            rows = await conn.fetch(
                """SELECT id, type, raw_text, summary, importance, recall_count,
                          last_recalled, timestamp, decay_score, archived
                   FROM episodes
                   WHERE archived = FALSE
                   ORDER BY timestamp DESC
                   LIMIT $1""",
                top_k,
            )
            return [_episode_row(r) for r in rows]

    async def bump_recall(self, id: str) -> None:
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE episodes
                   SET recall_count = recall_count + 1,
                       last_recalled = now()
                   WHERE id = $1""",
                id,
            )

    async def list_recent(self, hours: int = 24) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, type, raw_text, summary, importance, recall_count,
                          last_recalled, timestamp, decay_score, archived
                   FROM episodes
                   WHERE timestamp >= $1
                   ORDER BY timestamp DESC""",
                cutoff,
            )
        return [_episode_row(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Decision episodes (exact-match learning loop)
    # ------------------------------------------------------------------ #

    async def store_decision_episode(
        self,
        decision_id: str,
        intent_class: str,
        action_type: str,
        entity_class: str,
        actor_role: str,
        context_snapshot: dict[str, Any] | None = None,
        scores_json: str | None = None,
        generation_path: str = "MODEL",
    ) -> str:
        episode_id = str(uuid.uuid4())
        if not self._pool:
            return episode_id
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO decision_episodes
                     (episode_id, decision_id, intent_class, action_type, entity_class,
                      actor_role, context_snapshot, scores_json, generation_path)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                episode_id, decision_id, intent_class, action_type, entity_class,
                actor_role,
                _jsonb(context_snapshot) if context_snapshot else None,
                _jsonb(scores_json) if scores_json else None,
                generation_path,
            )
        return episode_id

    async def complete_decision_episode(
        self,
        decision_id: str,
        outcome: str,
        prediction_delta: float | None = None,
        early_reextraction_flag: bool = False,
    ) -> str | None:
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT episode_id FROM decision_episodes WHERE decision_id = $1",
                decision_id,
            )
            if not row:
                return None
            episode_id = row["episode_id"]
            await conn.execute(
                """UPDATE decision_episodes
                   SET outcome = $1, prediction_delta = $2,
                       early_reextraction_flag = $3, completed_at = now()
                   WHERE episode_id = $4""",
                outcome, prediction_delta, early_reextraction_flag, episode_id,
            )
        return episode_id

    async def fetch_decision_episodes_since(
        self,
        since: datetime,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT episode_id, decision_id, intent_class, action_type,
                          entity_class, actor_role, outcome, prediction_delta,
                          context_snapshot, scores_json, generation_path,
                          created_at, completed_at
                   FROM decision_episodes
                   WHERE created_at >= $1
                   ORDER BY created_at ASC
                   LIMIT $2""",
                since, limit,
            )
        return [_decision_row(r) for r in rows]

    async def fetch_decision_episodes_with_scores(
        self,
        since: datetime,
    ) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT episode_id, decision_id, intent_class, action_type,
                          entity_class, actor_role, outcome, prediction_delta,
                          scores_json, created_at, completed_at
                   FROM decision_episodes
                   WHERE created_at >= $1 AND outcome IS NOT NULL
                     AND scores_json IS NOT NULL
                   ORDER BY created_at DESC""",
                since,
            )
        return [
            {
                "episode_id": str(r["episode_id"]),
                "decision_id": r["decision_id"],
                "intent_class": r["intent_class"],
                "action_type": r["action_type"],
                "entity_class": r["entity_class"],
                "actor_role": r["actor_role"],
                "outcome": r["outcome"],
                "prediction_delta": r["prediction_delta"],
                "scores_json": r["scores_json"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "completed_at": r["completed_at"].isoformat() if r["completed_at"] else None,
            }
            for r in rows
        ]

    async def fetch_for_manifest(
        self,
        intent_class: str,
        entity_class: str,
        actor_role: str,
        lookback_days: int = 30,
        max_episodes: int = 20,
    ) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT episode_id, decision_id, action_type, entity_class, actor_role,
                          outcome, created_at, completed_at
                   FROM decision_episodes
                   WHERE intent_class = $1 AND entity_class = $2 AND actor_role = $3
                     AND created_at >= $4 AND outcome IS NOT NULL
                   ORDER BY created_at DESC
                   LIMIT $5""",
                intent_class, entity_class, actor_role, cutoff, max_episodes,
            )
        return [
            {
                "episode_id": str(r["episode_id"]),
                "decision_id": r["decision_id"],
                "action_type": r["action_type"],
                "entity_class": r["entity_class"],
                "outcome": r["outcome"],
                "created_at": r["created_at"].timestamp() if r["created_at"] else 0.0,
                "completed_at": r["completed_at"].timestamp() if r["completed_at"] else None,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # Patterns
    # ------------------------------------------------------------------ #

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
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO patterns
                     (pattern_id, context_signature, intent_class, action_type,
                      entity_class, actor_role, success_rate, confidence,
                      observation_count, avg_prediction_delta, extraction_run_id)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                   ON CONFLICT (context_signature) DO UPDATE SET
                     success_rate = EXCLUDED.success_rate,
                     confidence = EXCLUDED.confidence,
                     observation_count = EXCLUDED.observation_count,
                     avg_prediction_delta = EXCLUDED.avg_prediction_delta,
                     updated_at = now()""",
                str(uuid.uuid4()), context_signature, intent_class, action_type,
                entity_class, actor_role, success_rate, confidence,
                observation_count, avg_prediction_delta, extraction_run_id,
            )

    async def fetch_patterns_for_manifest(
        self,
        intent_class: str,
        entity_class: str,
        actor_role: str,
        max_patterns: int = 10,
    ) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT pattern_id, context_signature, intent_class, action_type,
                          entity_class, actor_role, success_rate, confidence,
                          observation_count, avg_prediction_delta
                   FROM patterns
                   WHERE intent_class = $1 AND entity_class = $2 AND actor_role = $3
                   ORDER BY confidence DESC
                   LIMIT $4""",
                intent_class, entity_class, actor_role, max_patterns,
            )
        return [dict(r) | {"pattern_id": str(r["pattern_id"])} for r in rows]

    async def fetch_patterns_low_success(
        self,
        max_success_rate: float = 0.4,
        min_confidence: float = 0.5,
    ) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT pattern_id, context_signature, intent_class, action_type,
                          entity_class, actor_role, success_rate, confidence,
                          observation_count, avg_prediction_delta
                   FROM patterns
                   WHERE success_rate <= $1 AND confidence >= $2
                   ORDER BY success_rate ASC""",
                max_success_rate, min_confidence,
            )
        return [dict(r) | {"pattern_id": str(r["pattern_id"])} for r in rows]

    async def fetch_all_patterns(self) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT pattern_id, context_signature, intent_class, action_type,
                          entity_class, actor_role, success_rate, confidence,
                          observation_count, avg_prediction_delta
                   FROM patterns
                   ORDER BY updated_at DESC"""
            )
        return [dict(r) | {"pattern_id": str(r["pattern_id"])} for r in rows]

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    async def execute(self, query: str, *args: Any) -> str:
        if not self._pool:
            return ""
        async with self._pool.acquire() as conn:
            await conn.execute(query, *args)
        return ""

    async def fetchval(self, query: str, *args: Any) -> Any:
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)


def _to_vector(embedding: list[float] | None) -> str | None:
    if embedding is None:
        return None
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _jsonb(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _episode_row(r: asyncpg.Record, sim: float | None = None) -> dict[str, Any]:
    ts = r["timestamp"]
    lr = r["last_recalled"]
    return {
        "id": str(r["id"]),
        "type": r["type"],
        "raw_text": r["raw_text"],
        "summary": r["summary"],
        "importance": float(r["importance"]),
        "recall_count": int(r["recall_count"]),
        "last_recalled": lr.isoformat() if lr else None,
        "timestamp": ts.isoformat() if ts else "",
        "decay_score": float(r["decay_score"]),
        "archived": bool(r["archived"]),
        "similarity": sim if sim is not None else 1.0,
    }


def _decision_row(r: asyncpg.Record) -> dict[str, Any]:
    return {
        "episode_id": str(r["episode_id"]),
        "decision_id": r["decision_id"],
        "intent_class": r["intent_class"],
        "action_type": r["action_type"],
        "entity_class": r["entity_class"],
        "actor_role": r["actor_role"],
        "outcome": r["outcome"],
        "prediction_delta": r["prediction_delta"],
        "context_snapshot": r["context_snapshot"],
        "scores_json": r["scores_json"],
        "generation_path": r["generation_path"],
        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        "completed_at": r["completed_at"].isoformat() if r["completed_at"] else None,
    }
