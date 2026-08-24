"""Agent dispatch run store — queue of tasks for external coding-agent runners.

Mirrors GoalStore/WorkflowStore conventions: aiosqlite, dict rows, per-call
connections. The claim/lease protocol is the same one the workflow executor
uses: claiming marks RUNNING with a lease; expired leases are re-claimable;
only RUNNING rows can be completed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

from .session_ingest.redactor import redact_text


def _now() -> float:
    return time.time()


class AgentRunStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    async def create_run(
        self, *, prompt: str, workspace: str | None = None,
        approval_id: str | None = None,
    ) -> dict[str, Any]:
        run_id = str(uuid4())
        now = _now()
        if workspace is None:
            workspace = f"~/xnch-agents/{run_id}"
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO agent_runs (id, status, prompt, workspace,"
                " approval_id, created_at, updated_at)"
                " VALUES (?, 'QUEUED', ?, ?, ?, ?, ?)",
                (run_id, prompt, workspace, approval_id, now, now),
            )
            await db.commit()
        row = await self.get_run(run_id)
        assert row is not None
        return row

    async def claim_next(self, runner_id: str, ttl_s: int = 1800) -> dict[str, Any] | None:
        """Claim the oldest QUEUED run — or re-claim an expired-lease RUNNING one."""
        now = _now()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM agent_runs"
                " WHERE status = 'QUEUED'"
                "    OR (status = 'RUNNING' AND (lease_expires_at IS NULL"
                "                                OR lease_expires_at < ?))"
                " ORDER BY created_at ASC LIMIT 1",
                (now,),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                return None
            claimed = dict(row)
            await db.execute(
                "UPDATE agent_runs SET status = 'RUNNING', runner_id = ?,"
                " lease_expires_at = ?, updated_at = ? WHERE id = ?",
                (runner_id, now + ttl_s, now, claimed["id"]),
            )
            await db.commit()
        return await self.get_run(str(claimed["id"]))

    async def complete_run(
        self,
        run_id: str,
        *,
        outcome_status: str,
        exit_code: int | None = None,
        output_path: str | None = None,
        error: str | None = None,
        result_text: str | None = None,
    ) -> dict[str, Any] | None:
        if outcome_status not in ("DONE", "FAILED"):
            raise ValueError(f"invalid outcome_status: {outcome_status!r}")
        if result_text:
            result_text, _ = redact_text(result_text)
        if error:
            error, _ = redact_text(error)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT status FROM agent_runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
            if row is None or row["status"] != "RUNNING":
                return None
            await db.execute(
                "UPDATE agent_runs SET status = ?, exit_code = ?, output_path = ?,"
                " error = ?, result_text = ?, lease_expires_at = NULL,"
                " updated_at = ? WHERE id = ?",
                (outcome_status, exit_code, output_path, error, result_text,
                 _now(), run_id),
            )
            await db.commit()
        return await self.get_run(run_id)

    async def list_runs(
        self, status: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM agent_runs"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
        return dict(row) if row else None

    async def active_run_for_approval(self, approval_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM agent_runs WHERE approval_id = ?"
                " AND status IN ('QUEUED','RUNNING')"
                " ORDER BY created_at DESC LIMIT 1",
                (approval_id,),
            ) as cur:
                row = await cur.fetchone()
        return dict(row) if row else None

    async def get_run_by_approval(self, approval_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM agent_runs WHERE approval_id = ?"
                " ORDER BY created_at DESC LIMIT 1",
                (approval_id,),
            ) as cur:
                row = await cur.fetchone()
        return dict(row) if row else None
