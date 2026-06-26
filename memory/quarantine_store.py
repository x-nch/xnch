from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import asyncpg


class QuarantineStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=2)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def quarantine(
        self,
        memory_type: str,
        raw_text: str | None,
        summary: str | None,
        quarantine_reason: str,
        quarantined_by: str,
        original_actor_role: str,
        original_trust_level: str,
        importance: float = 1.0,
    ) -> str:
        if not self._pool:
            return ""
        qid = str(uuid4())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO quarantine_memories
                   (id, memory_type, raw_text, summary, importance,
                    quarantine_reason, quarantined_by, original_actor_role, original_trust_level)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                qid, memory_type, raw_text, summary, importance,
                quarantine_reason, quarantined_by, original_actor_role, original_trust_level,
            )
        return qid

    async def release_to_memory(self, id: str, released_by: str) -> bool:
        if not self._pool:
            return False
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE quarantine_memories
                   SET released_at = now(), released_by = $1
                   WHERE id = $2::uuid AND released_at IS NULL""",
                released_by, id,
            )
            return result != "UPDATE 0"

    async def list_quarantined(self) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, memory_type, raw_text, summary, importance,
                          quarantine_reason, quarantined_by, original_actor_role,
                          original_trust_level, created_at, released_at, released_by
                   FROM quarantine_memories
                   WHERE released_at IS NULL
                   ORDER BY created_at DESC"""
            )
        return [dict(r) for r in rows]
