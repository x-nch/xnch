"""WorkflowStore + approval queue persistence.

Mirrors GoalStore conventions: aiosqlite, dict rows, JSON-as-string columns,
epoch-second timestamps. v1 embeds run steps as JSON on ``workflow_runs``
(see docs/superpowers/specs/2026-08-22-workflows-backend-design.md §2); the
``approvals`` table is first-class and producer-agnostic from day one.

This module imports only stdlib + aiosqlite so it is loadable in minimal
environments.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

from xnch.models.workflow import ELEVATED_KINDS

_TERMINAL_STEP = frozenset({"DONE", "REJECTED", "EXPIRED", "CANCELLED"})


def _now() -> float:
    return time.time()


class ApprovalConflict(Exception):
    """Approval is not in AWAITING_APPROVAL (already decided/expired)."""


class WorkflowNotFound(Exception):
    pass


class WorkflowStore:
    def __init__(self, db_path: Path, executor_enabled: bool = False) -> None:
        self._db = db_path
        # False (v1): approve ⇒ step DONE immediately (no executor deployed).
        # True (P2):  approve ⇒ APPROVED; nexi executor claims and executes.
        self._executor_enabled = executor_enabled

    # ------------------------------------------------------------------
    # Workflow definitions
    # ------------------------------------------------------------------

    async def create_workflow(
        self,
        *,
        owner_actor_id: str,
        name: str,
        description: str | None,
        trigger: dict[str, Any],
        steps: list[dict[str, Any]],
    ) -> str:
        wf_id = str(uuid4())
        now = _now()
        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                "INSERT INTO workflows (id, owner_actor_id, name, description,"
                " trigger_json, steps_json, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    wf_id,
                    owner_actor_id,
                    name,
                    description,
                    json.dumps(trigger),
                    json.dumps(steps),
                    now,
                    now,
                ),
            )
            await db.commit()
        return wf_id

    async def get_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM workflows WHERE id = ?", (workflow_id,)
            ) as cur:
                row = await cur.fetchone()
        return dict(row) if row else None

    async def list_workflows(self) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM workflows ORDER BY created_at DESC"
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def update_workflow(
        self, workflow_id: str, patch: dict[str, Any]
    ) -> dict[str, Any] | None:
        existing = await self.get_workflow(workflow_id)
        if not existing:
            return None
        sets: list[str] = ["updated_at = ?"]
        params: list[Any] = [_now()]
        for col, key in (
            ("name", "name"),
            ("description", "description"),
            ("trigger_json", "trigger"),
            ("steps_json", "steps"),
        ):
            if key in patch and patch[key] is not None:
                value = patch[key]
                if key in ("trigger", "steps"):
                    value = json.dumps(value)
                sets.append(f"{col} = ?")
                params.append(value)
        params.append(workflow_id)
        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                f"UPDATE workflows SET {', '.join(sets)} WHERE id = ?", params
            )
            await db.commit()
        return await self.get_workflow(workflow_id)

    async def delete_workflow(self, workflow_id: str) -> bool:
        async with aiosqlite.connect(self._db) as db:
            cur = await db.execute(
                "DELETE FROM workflows WHERE id = ?", (workflow_id,)
            )
            await db.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Runs — expand steps, create approvals for gated steps
    # ------------------------------------------------------------------

    async def create_run(
        self,
        *,
        workflow_id: str,
        actor: str,
        trigger: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        approval_ttl_s: float = 3600.0,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Create a run. Returns ``(run_row_or_existing, created)``.

        If ``idempotency_key`` was seen before, returns the existing run with
        ``created=False`` and performs no writes.
        """
        if idempotency_key:
            async with aiosqlite.connect(self._db) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM workflow_runs WHERE idempotency_key = ?",
                    (idempotency_key,),
                ) as cur:
                    row = await cur.fetchone()
            if row:
                return dict(row), False

        wf = await self.get_workflow(workflow_id)
        if not wf:
            raise WorkflowNotFound(workflow_id)

        now = _now()
        run_id = str(uuid4())
        step_defs: list[dict[str, Any]] = json.loads(wf["steps_json"] or "[]")
        trigger_json = json.dumps(trigger or json.loads(wf["trigger_json"] or "{}"))

        run_steps: list[dict[str, Any]] = []
        approvals: list[tuple[Any, ...]] = []
        for i, sd in enumerate(step_defs):
            step_uuid = str(uuid4())
            gated = bool(sd.get("requires_approval", True))
            status = "AWAITING_APPROVAL" if gated else "DONE"
            approval_id: str | None = None
            if gated:
                approval_id = str(uuid4())
                risk = "elevated" if sd.get("kind") in ELEVATED_KINDS else "low"
                payload = {
                    "run_id": run_id,
                    "workflow_id": workflow_id,
                    "workflow_name": wf["name"],
                    "step_index": i,
                    "kind": sd.get("kind", "other"),
                    "summary": sd.get("summary", ""),
                    "target": sd.get("target"),
                    "args": sd.get("args"),
                    "preview": sd.get("preview"),
                }
                approvals.append(
                    (
                        approval_id,
                        "workflow_step",
                        step_uuid,
                        json.dumps(payload),
                        "AWAITING_APPROVAL",
                        risk,
                        now + approval_ttl_s,
                        now,
                    )
                )
            run_steps.append(
                {
                    "step_uuid": step_uuid,
                    "index": i,
                    "kind": sd.get("kind", "other"),
                    "summary": sd.get("summary", ""),
                    "target": sd.get("target"),
                    "args": sd.get("args"),
                    "preview": sd.get("preview"),
                    "requires_approval": gated,
                    "status": status,
                    "approval_id": approval_id,
                    "retry_count": 0,
                    "max_retries": 3,
                    "next_retry_at": None,
                }
            )

        all_resolved = all(rs["status"] in _TERMINAL_STEP for rs in run_steps)
        run_status = "COMPLETED" if all_resolved else "RUNNING"

        step_rows: list[tuple[Any, ...]] = [
            (
                rs["step_uuid"],
                run_id,
                rs["index"],
                rs["kind"],
                rs["summary"],
                json.dumps(
                    {
                        "target": rs.get("target"),
                        "args": rs.get("args"),
                        "preview": rs.get("preview"),
                    }
                ),
                1 if rs["requires_approval"] else 0,
                rs["status"],
                rs["approval_id"],
                now,
            )
            for rs in run_steps
        ]

        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                "INSERT INTO workflow_runs (id, workflow_id, status,"
                " trigger_json, steps_json, idempotency_key, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    workflow_id,
                    run_status,
                    trigger_json,
                    json.dumps(run_steps),
                    idempotency_key,
                    now,
                    now,
                ),
            )
            if step_rows:
                await db.executemany(
                    "INSERT INTO workflow_run_steps (step_uuid, run_id, idx, kind,"
                    " summary, payload_json, requires_approval, status,"
                    " approval_id, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    step_rows,
                )
            await db.executemany(
                "INSERT INTO approvals (id, producer_type, producer_id,"
                " payload_json, status, risk_class, expires_at, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                approvals,
            )
            events = [
                (
                    rs["step_uuid"],
                    "RUN_CREATED"
                    if rs["status"] == "AWAITING_APPROVAL"
                    else "AUTO_DONE",
                    actor,
                    now,
                    json.dumps(rs),
                )
                for rs in run_steps
            ]
            await db.executemany(
                "INSERT INTO step_events (step_uuid, event_type, actor, ts,"
                " snapshot_json) VALUES (?, ?, ?, ?, ?)",
                events,
            )
            await db.commit()

        run = await self.get_run(run_id)
        assert run is not None
        return run, True

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM workflow_runs WHERE id = ?", (run_id,)
            ) as cur:
                row = await cur.fetchone()
        return dict(row) if row else None

    async def list_runs(
        self,
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM workflow_runs"
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if workflow_id:
            clauses.append("workflow_id = ?")
            params.append(workflow_id)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(q, params) as cur:
                return [dict(r) for r in await cur.fetchall()]

    # ------------------------------------------------------------------
    # Approvals
    # ------------------------------------------------------------------

    async def _lazy_expire(self, db: aiosqlite.Connection) -> int:
        """Flip past-due AWAITING_APPROVAL rows to EXPIRED. Returns count."""
        now = _now()
        async with db.execute(
            "SELECT id, producer_id, payload_json FROM approvals"
            " WHERE status = 'AWAITING_APPROVAL' AND expires_at IS NOT NULL"
            " AND expires_at < ?",
            (now,),
        ) as cur:
            rows = [dict(zip(("id", "producer_id", "payload_json"), r)) for r in await cur.fetchall()]
        if not rows:
            return 0
        for row in rows:
            await db.execute(
                "UPDATE approvals SET status = 'EXPIRED' WHERE id = ?", (row["id"],)
            )
            await db.execute(
                "INSERT INTO step_events (step_uuid, event_type, actor, ts,"
                " snapshot_json) VALUES (?, 'EXPIRED', 'system', ?, ?)",
                (row["producer_id"], _now(), row["payload_json"]),
            )
            await self._apply_step_status(db, row["producer_id"], "EXPIRED")
        return len(rows)

    async def list_approvals(
        self,
        *,
        status: str | None = "pending",
        producer_type: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            expired = await self._lazy_expire(db)
            if expired:
                await db.commit()
            q = "SELECT * FROM approvals"
            clauses: list[str] = []
            params: list[Any] = []
            if status == "pending":
                clauses.append("status = 'AWAITING_APPROVAL'")
            elif status:
                clauses.append("status = ?")
                params.append(status)
            if producer_type:
                clauses.append("producer_type = ?")
                params.append(producer_type)
            if clauses:
                q += " WHERE " + " AND ".join(clauses)
            q += " ORDER BY created_at ASC LIMIT ?"
            params.append(limit)
            async with db.execute(q, params) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM approvals WHERE id = ?", (approval_id,)
            ) as cur:
                row = await cur.fetchone()
        return dict(row) if row else None

    async def decide_approval(
        self,
        approval_id: str,
        *,
        decision: str,
        actor: str,
        note: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Approve/reject an AWAITING_APPROVAL row.

        Raises :class:`ApprovalConflict` when already decided/expired/cancelled.
        """
        if decision not in ("approve", "reject"):
            raise ValueError("decision must be approve|reject")
        if idempotency_key:
            async with aiosqlite.connect(self._db) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT id FROM approvals WHERE idempotency_key = ?",
                    (idempotency_key,),
                ) as cur:
                    seen = await cur.fetchone()
            if seen:
                existing = await self.get_approval(str(seen["id"]))
                assert existing is not None
                return existing

        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM approvals WHERE id = ?", (approval_id,)
            ) as cur:
                row = await cur.fetchone()
            if not row:
                raise WorkflowNotFound(approval_id)
            appr = dict(row)

            if (
                appr["status"] == "AWAITING_APPROVAL"
                and appr["expires_at"] is not None
                and appr["expires_at"] < _now()
            ):
                await db.execute(
                    "UPDATE approvals SET status='EXPIRED' WHERE id=?",
                    (approval_id,),
                )
                await db.execute(
                    "INSERT INTO step_events (step_uuid, event_type, actor, ts,"
                    " snapshot_json) VALUES (?, 'EXPIRED', 'system', ?, ?)",
                    (appr["producer_id"], _now(), appr["payload_json"]),
                )
                await self._apply_step_status(db, appr["producer_id"], "EXPIRED")
                await db.commit()
                raise ApprovalConflict("approval expired")

            if appr["status"] != "AWAITING_APPROVAL":
                raise ApprovalConflict(f"approval is {appr['status']}")

            new_status = "APPROVED" if decision == "approve" else "REJECTED"
            now = _now()
            await db.execute(
                "UPDATE approvals SET status=?, decision_note=?, decided_by=?,"
                " decided_at=?, idempotency_key=? WHERE id=?",
                (
                    new_status,
                    note,
                    actor,
                    now,
                    idempotency_key,
                    approval_id,
                ),
            )
            await db.execute(
                "INSERT INTO step_events (step_uuid, event_type, actor, ts,"
                " snapshot_json) VALUES (?, ?, ?, ?, ?)",
                (
                    appr["producer_id"],
                    new_status,
                    actor,
                    now,
                    json.dumps({**appr, "note": note}),
                ),
            )
            if decision == "approve":
                step_status = "APPROVED" if self._executor_enabled else "DONE"
            else:
                step_status = "REJECTED"
            await self._apply_step_status(db, appr["producer_id"], step_status)
            await db.commit()

        updated = await self.get_approval(approval_id)
        assert updated is not None
        return updated

    async def cancel_approvals_for_run(self, run_id: str, actor: str) -> int:
        """Cancel every still-pending approval belonging to a run."""
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id, producer_id FROM approvals WHERE status ="
                " 'AWAITING_APPROVAL' AND json_extract(payload_json, '$.run_id') = ?",
                (run_id,),
            ) as cur:
                rows = [dict(r) for r in await cur.fetchall()]
            for row in rows:
                await db.execute(
                    "UPDATE approvals SET status='CANCELLED', decided_by=?,"
                    " decided_at=? WHERE id=?",
                    (actor, _now(), row["id"]),
                )
                await db.execute(
                    "INSERT INTO step_events (step_uuid, event_type, actor, ts,"
                    " snapshot_json) VALUES (?, 'CANCELLED', ?, ?, '{}')",
                    (row["producer_id"], actor, _now()),
                )
                await self._apply_step_status(db, row["producer_id"], "CANCELLED")
            if rows:
                await db.commit()
            return len(rows)

    # ------------------------------------------------------------------

    async def _apply_step_status(
        self, db: aiosqlite.Connection, step_uuid: str, new_status: str
    ) -> None:
        """Update a step's status in the row-level table (P2) and mirror into
        the run's steps_json snapshot; roll up run status from rows when they
        exist, else from JSON (legacy runs)."""
        now = _now()
        async with db.execute(
            "SELECT run_id FROM workflow_run_steps WHERE step_uuid = ?",
            (step_uuid,),
        ) as cur:
            row = await cur.fetchone()

        if row:
            run_id = row[0]
            await db.execute(
                "UPDATE workflow_run_steps SET status=?, updated_at=?"
                " WHERE step_uuid=?",
                (new_status, now, step_uuid),
            )
            statuses: list[str] = [
                r[0]
                for r in await (
                    await db.execute(
                        "SELECT status FROM workflow_run_steps WHERE run_id=?"
                        " ORDER BY idx",
                        (run_id,),
                    )
                ).fetchall()
            ]
        else:
            # legacy path — locate run via JSON snapshot
            async with db.execute(
                "SELECT id, steps_json FROM workflow_runs WHERE steps_json LIKE ?",
                (f'%"{step_uuid}"%',),
            ) as cur:
                found = await cur.fetchone()
            if not found:
                return
            run_id, steps_raw = found[0], found[1]
            steps: list[dict[str, Any]] = json.loads(steps_raw or "[]")
            for rs in steps:
                if rs.get("step_uuid") == step_uuid:
                    rs["status"] = new_status
            await db.execute(
                "UPDATE workflow_runs SET steps_json=? WHERE id=?",
                (json.dumps(steps), run_id),
            )
            statuses = [s.get("status", "") for s in steps]

        any_hard_fail = any(s in ("REJECTED", "FAILED") for s in statuses)
        all_terminal = all(s in _TERMINAL_STEP for s in statuses)
        run_status = (
            "FAILED"
            if any_hard_fail
            else ("COMPLETED" if all_terminal else "RUNNING")
        )
        await db.execute(
            "UPDATE workflow_runs SET status=?, updated_at=? WHERE id=?",
            (run_status, now, run_id),
        )
        # keep JSON snapshot's changed member in sync
        if row:
            async with db.execute(
                "SELECT steps_json FROM workflow_runs WHERE id=?", (run_id,)
            ) as cur:
                raw = (await cur.fetchone())[0]
            steps: list[dict[str, Any]] = json.loads(raw or "[]")
            for rs in steps:
                if rs.get("step_uuid") == step_uuid:
                    rs["status"] = new_status
            await db.execute(
                "UPDATE workflow_runs SET steps_json=? WHERE id=?",
                (json.dumps(steps), run_id),
            )

    # ------------------------------------------------------------------
    # Executor claims + outcomes (P2)
    # ------------------------------------------------------------------

    _TERMINAL_STEP = frozenset({"DONE", "REJECTED", "EXPIRED", "CANCELLED", "FAILED"})
    RETRY_BACKOFF_BASE_S = 30.0

    async def get_run_step_rows(self, run_id: str) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM workflow_run_steps WHERE run_id=? ORDER BY idx",
                (run_id,),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def claim_next_approved_step(
        self, *, lease_owner: str, ttl_s: int = 120
    ) -> dict[str, Any] | None:
        """Atomically claim one APPROVED (or due-RETRYING / stale-CLAIMED) step."""
        now = _now()
        expires = now + ttl_s
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                UPDATE workflow_run_steps SET
                    status='CLAIMED', lease_owner=?, lease_expires_at=?, updated_at=?
                WHERE step_uuid = (
                    SELECT step_uuid FROM workflow_run_steps
                    WHERE (status='APPROVED')
                       OR (status='RETRYING' AND next_retry_at IS NOT NULL
                           AND next_retry_at <= ?)
                       OR (status='CLAIMED' AND lease_expires_at IS NOT NULL
                           AND lease_expires_at < ?)
                    ORDER BY created_at ASC LIMIT 1
                )
                RETURNING *
                """,
                (lease_owner, expires, now, now, now),
            ) as cur:
                claimed = await cur.fetchone()
            if not claimed:
                return None
            row = dict(claimed)
            await db.execute(
                "INSERT INTO step_events (step_uuid, event_type, actor, ts,"
                " snapshot_json) VALUES (?, 'CLAIMED', ?, ?, ?)",
                (row["step_uuid"], lease_owner, now, json.dumps(row)),
            )
            await db.commit()
        return row

    async def complete_step(
        self, step_uuid: str, *, outcome_status: str, actor: str
    ) -> dict[str, Any]:
        """Apply an executor outcome. SUCCESS/PARTIAL ⇒ DONE;
        FAILURE ⇒ RETRYING (with backoff) until max_retries, then FAILED."""
        if outcome_status not in ("SUCCESS", "PARTIAL", "FAILURE"):
            raise ValueError("outcome_status must be SUCCESS|PARTIAL|FAILURE")
        now = _now()
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM workflow_run_steps WHERE step_uuid=?", (step_uuid,)
            ) as cur:
                row = await cur.fetchone()
            if not row:
                raise WorkflowNotFound(step_uuid)
            step = dict(row)

            if outcome_status in ("SUCCESS", "PARTIAL"):
                new_status = "DONE"
                retry_count = step["retry_count"]
                next_retry_at = None
            else:
                retry_count = step["retry_count"] + 1
                if retry_count < step["max_retries"]:
                    new_status = "RETRYING"
                    next_retry_at = now + self.RETRY_BACKOFF_BASE_S * (
                        2 ** (retry_count - 1)
                    )
                else:
                    new_status = "FAILED"
                    next_retry_at = None

            await db.execute(
                "UPDATE workflow_run_steps SET status=?, retry_count=?,"
                " next_retry_at=?, lease_owner=NULL, lease_expires_at=NULL,"
                " updated_at=? WHERE step_uuid=?",
                (new_status, retry_count, next_retry_at, now, step_uuid),
            )
            event = dict(step)
            event.update(
                {
                    "outcome_status": outcome_status,
                    "status": new_status,
                    "retry_count": retry_count,
                    "next_retry_at": next_retry_at,
                }
            )
            await db.execute(
                "INSERT INTO step_events (step_uuid, event_type, actor, ts,"
                " snapshot_json) VALUES (?, ?, ?, ?, ?)",
                (step_uuid, new_status, actor, now, json.dumps(event)),
            )
            await self._apply_step_status(db, step_uuid, new_status)
            await db.commit()

        rows = await self.get_run_step_rows(step["run_id"])
        for r in rows:
            if r["step_uuid"] == step_uuid:
                return r
        raise WorkflowNotFound(step_uuid)

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    async def step_events(self, step_uuid: str) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT seq, step_uuid, event_type, actor, ts, snapshot_json"
                " FROM step_events WHERE step_uuid = ? ORDER BY seq",
                (step_uuid,),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]
