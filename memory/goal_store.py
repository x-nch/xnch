"""Goal Store — durable, resumable goal state driving the agentic loop."""
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

_ELIGIBLE = ("PENDING", "ACTIVE")
_TERMINAL = ("COMPLETED", "FAILED", "CANCELLED")


class GoalStore:
    def __init__(self, db_path: Path) -> None:
        self._db = db_path

    async def create_goal(
        self, *, owner_actor_id: str, objective: str,
        max_steps: int = 10, failure_threshold: int = 3,
        simulation_plan: list[dict[str, Any]] | None = None,
    ) -> str:
        goal_id = str(uuid4())
        now = time.time()
        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                "INSERT INTO goals (goal_id, owner_actor_id, objective, status,"
                " max_steps, failure_threshold, next_due_at, simulation_plan,"
                " created_at, updated_at)"
                " VALUES (?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?)",
                (goal_id, owner_actor_id, objective, max_steps, failure_threshold,
                 now, json.dumps(simulation_plan or []), now, now))
            await db.commit()
        return goal_id

    async def get_goal(self, goal_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM goals WHERE goal_id = ?", (goal_id,)) as cur:
                row = await cur.fetchone()
        return dict(row) if row else None

    async def list_goals(self, status: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM goals" + (" WHERE status = ?" if status else "") + " ORDER BY created_at"
        params = (status,) if status else ()
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(q, params) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def claim_next_goal(self, lease_owner: str, lease_ttl_s: int = 120) -> dict[str, Any] | None:
        now = time.time()
        expires = now + lease_ttl_s
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                "UPDATE goals SET status='ACTIVE', lease_owner=NULL, lease_expires_at=NULL"
                " WHERE status='RUNNING' AND lease_expires_at < ?", (now,))
            async with db.execute(
                "UPDATE goals SET status='RUNNING', lease_owner=?, lease_expires_at=?, updated_at=?"
                " WHERE goal_id = (SELECT goal_id FROM goals WHERE status IN ('PENDING','ACTIVE')"
                " AND next_due_at <= ? ORDER BY next_due_at ASC LIMIT 1) RETURNING *",
                (lease_owner, expires, now, now)) as cur:
                row = await cur.fetchone()
            await db.commit()
        return dict(row) if row else None

    async def complete_step(self, goal_id: str, outcome_status: str) -> dict[str, Any] | None:
        now = time.time()
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM goals WHERE goal_id = ?", (goal_id,)) as cur:
                row = await cur.fetchone()
            if not row:
                return None
            g = dict(row)
            steps = g["steps_completed"] + 1
            consecutive = g["consecutive_failures"]
            if outcome_status in ("SUCCESS", "PARTIAL"):
                consecutive = 0
                status = "COMPLETED" if steps >= g["max_steps"] else "ACTIVE"
            elif outcome_status == "FAILURE":
                consecutive += 1
                status = "FAILED" if consecutive >= g["failure_threshold"] else "ACTIVE"
            else:
                status = g["status"]
            progress = f"{g['progress']}\nstep {steps}: {outcome_status}".strip()
            await db.execute(
                "UPDATE goals SET status=?, progress=?, steps_completed=?, consecutive_failures=?,"
                " last_step_outcome=?, next_due_at=?, lease_owner=NULL, lease_expires_at=NULL,"
                " updated_at=? WHERE goal_id=?",
                (status, progress, steps, consecutive, outcome_status,
                 None if status in _TERMINAL else now, now, goal_id))
            await db.commit()
        return await self.get_goal(goal_id)

    async def update_goal(self, goal_id: str, *, status: str | None = None,
                          progress: str | None = None) -> dict[str, Any] | None:
        now = time.time()
        async with aiosqlite.connect(self._db) as db:
            if status is not None:
                await db.execute(
                    "UPDATE goals SET status=?, updated_at=?, lease_owner=NULL,"
                    " lease_expires_at=NULL WHERE goal_id=?", (status, now, goal_id))
            if progress is not None:
                await db.execute("UPDATE goals SET progress=?, updated_at=? WHERE goal_id=?",
                                 (progress, now, goal_id))
            await db.commit()
        return await self.get_goal(goal_id)
