"""Actor → role + capability_set resolution from the governance store."""
import json
from pathlib import Path
from typing import Any

import aiosqlite

from ..config import settings


_BOOTSTRAP_ACTORS = [
    {"actor_id": "admin", "role": "ADMIN", "capability_set": ["DEPLOY", "READ", "QUERY", "ADMIN", "SCHEMA_WRITE"]},
    {"actor_id": "operator", "role": "OPERATOR", "capability_set": ["DEPLOY", "READ", "QUERY"]},
    {"actor_id": "viewer", "role": "VIEWER", "capability_set": ["READ", "QUERY"]},
    {"actor_id": "agent", "role": "AGENT", "capability_set": ["READ", "QUERY", "DEPLOY"]},
]


class Actor:
    def __init__(self, actor_id: str, role: str, capability_set: list[str]) -> None:
        self.id = actor_id
        self.role = role
        self.capability_set = capability_set

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "role": self.role, "capability_set": self.capability_set}


class GovernanceStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def resolve_actor(self, actor_id: str) -> Actor | None:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT actor_id, role, capability_set FROM actors WHERE actor_id = ?",
                (actor_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if not row:
            return None
        return Actor(row[0], row[1], json.loads(row[2]))

    async def upsert_actor(self, actor_id: str, role: str, capability_set: list[str]) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO actors (actor_id, role, capability_set, created_at, updated_at)
                   VALUES (?, ?, ?, unixepoch(), unixepoch())
                   ON CONFLICT(actor_id) DO UPDATE SET
                     role=excluded.role,
                     capability_set=excluded.capability_set,
                     updated_at=excluded.updated_at""",
                (actor_id, role, json.dumps(capability_set)),
            )
            await db.commit()

    async def bootstrap(self) -> None:
        """Insert default actors if the table is empty."""
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM actors") as cursor:
                count = (await cursor.fetchone())[0]
        if count == 0:
            for a in _BOOTSTRAP_ACTORS:
                await self.upsert_actor(a["actor_id"], a["role"], a["capability_set"])
