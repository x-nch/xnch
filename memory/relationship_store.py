"""Relationship memory store — PostgreSQL-backed entity relationship tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg


@dataclass
class RelationshipRecord:
    entity_a_id: str
    entity_b_id: str
    relationship_type: str
    strength: float
    reinforcement_count: int


class RelationshipStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def upsert_relationship(
        self,
        entity_a: str,
        entity_b: str,
        rel_type: str,
        evidence: str,
        strength: float = 1.0,
    ) -> None:
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO relationship_memory
                       (entity_a_id, entity_b_id, relationship_type, strength, evidence)
                   VALUES ($1, $2, $3, $4, ARRAY[$5])
                   ON CONFLICT (entity_a_id, entity_b_id, relationship_type)
                   DO UPDATE SET
                       strength = $4,
                       evidence = array_append(relationship_memory.evidence, $5),
                       reinforcement_count = relationship_memory.reinforcement_count + 1,
                       last_reinforced = now()""",
                entity_a,
                entity_b,
                rel_type,
                strength,
                evidence,
            )

    async def get_relationships(
        self, entity_id: str
    ) -> list[RelationshipRecord]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT entity_a_id, entity_b_id, relationship_type,
                          strength, reinforcement_count
                   FROM relationship_memory
                   WHERE entity_a_id = $1 OR entity_b_id = $1
                   ORDER BY strength DESC""",
                entity_id,
            )
        return [
            RelationshipRecord(
                entity_a_id=r["entity_a_id"],
                entity_b_id=r["entity_b_id"],
                relationship_type=r["relationship_type"],
                strength=r["strength"],
                reinforcement_count=r["reinforcement_count"],
            )
            for r in rows
        ]

    async def get_relationship_strength(
        self, entity_a: str, entity_b: str
    ) -> float | None:
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT strength FROM relationship_memory
                   WHERE entity_a_id = $1
                     AND entity_b_id = $2
                   LIMIT 1""",
                entity_a,
                entity_b,
            )
        return row["strength"] if row else None
