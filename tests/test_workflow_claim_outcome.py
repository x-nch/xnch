"""P2 executor infrastructure — row-level claims, leases, outcomes, and the
workflow_executor_enabled semantic switch.

Fails until: workflow_run_steps table exists, store gains claim_next_approved_step/
complete_step, decide_approval honors executor_enabled, and the claim/outcome
routes are wired.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

XNCH_ROOT = Path(__file__).resolve().parent.parent
if str(XNCH_ROOT) not in sys.path:
    sys.path.insert(0, str(XNCH_ROOT))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, XNCH_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_db = _load("xnch_p2_db", "memory/db.py")
_store_mod = _load("xnch_p2_store", "memory/workflow_store.py")
init_db = _db.init_db
WorkflowStore = _store_mod.WorkflowStore

STEPS = [
    {"id": "s1", "kind": "exec_tool", "summary": "Search",
     "target": "web_search", "requires_approval": True},
]


async def _mk(tmp_path, **kw):
    db_path = tmp_path / "p2.db"
    await init_db(db_path)
    store = WorkflowStore(db_path, **kw)
    wf_id = await store.create_workflow(
        owner_actor_id="op", name="W", description=None,
        trigger={"kind": "manual"}, steps=STEPS,
    )
    return store, wf_id


async def _approved_step(store, wf_id, *, executor_enabled):
    run, _ = await store.create_run(workflow_id=wf_id, actor="op")
    (approval,) = await store.list_approvals(status="pending")
    await store.decide_approval(approval["id"], decision="approve", actor="op")
    return run, approval


# ----------------------------------------------------------------------
# Rows table
# ----------------------------------------------------------------------


async def test_create_run_writes_row_level_steps(tmp_path):
    store, wf_id = await _mk(tmp_path)
    run, _ = await store.create_run(workflow_id=wf_id, actor="op")
    rows = await store.get_run_step_rows(run["id"])
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run["id"]
    assert row["status"] == "AWAITING_APPROVAL"
    assert row["approval_id"] is not None
    assert row["lease_owner"] is None and row["lease_expires_at"] is None


async def test_get_run_step_rows_empty_for_unknown_run(tmp_path):
    store, _ = await _mk(tmp_path)
    assert await store.get_run_step_rows("nope") == []


# ----------------------------------------------------------------------
# Executor-enabled semantics: approve leaves step APPROVED (non-terminal)
# ----------------------------------------------------------------------


async def test_executor_enabled_approve_marks_step_approved_not_done(tmp_path):
    store, wf_id = await _mk(tmp_path, executor_enabled=True)
    run, approval = await _approved_step(store, wf_id, executor_enabled=True)

    appr = await store.get_approval(approval["id"])
    assert appr["status"] == "APPROVED"

    rows = await store.get_run_step_rows(run["id"])
    assert rows[0]["status"] == "APPROVED"  # waiting for executor claim
    # run NOT completed yet
    assert (await store.get_run(run["id"]))["status"] == "RUNNING"


async def test_executor_disabled_approve_marks_done_v1(tmp_path):
    store, wf_id = await _mk(tmp_path)  # default off
    run, _ = await _approved_step(store, wf_id, executor_enabled=False)
    rows = await store.get_run_step_rows(run["id"])
    assert rows[0]["status"] == "DONE"
    assert (await store.get_run(run["id"]))["status"] == "COMPLETED"


# ----------------------------------------------------------------------
# Atomic claims with leases
# ----------------------------------------------------------------------


async def test_claim_next_approved_step_claims_once(tmp_path):
    store, wf_id = await _mk(tmp_path, executor_enabled=True)
    run, _ = await _approved_step(store, wf_id, executor_enabled=True)

    claimed = await store.claim_next_approved_step(lease_owner="nexi-1", ttl_s=120)
    assert claimed is not None
    assert claimed["status"] == "CLAIMED"
    assert claimed["lease_owner"] == "nexi-1"
    assert claimed["lease_expires_at"] is not None
    assert claimed["step_uuid"]

    # second claimer gets nothing (already leased)
    assert await store.claim_next_approved_step(lease_owner="nexi-2", ttl_s=120) is None


async def test_expired_lease_is_reclaimable(tmp_path):
    store, wf_id = await _mk(tmp_path, executor_enabled=True)
    await _approved_step(store, wf_id, executor_enabled=True)
    await store.claim_next_approved_step(lease_owner="nexi-1", ttl_s=-1)  # instantly stale
    reclaimed = await store.claim_next_approved_step(lease_owner="nexi-2", ttl_s=60)
    assert reclaimed is not None
    assert reclaimed["lease_owner"] == "nexi-2"


async def test_claim_skips_non_approved_and_future_retries(tmp_path):
    store, wf_id = await _mk(tmp_path, executor_enabled=True)
    run, approval = await _approved_step(store, wf_id, executor_enabled=True)
    # a RETRYING step with future next_retry_at must not be claimed
    await store.get_run_step_rows(run["id"])
    # simulate: manually put row into RETRYING with future retry time
    import aiosqlite
    async with aiosqlite.connect(store._db) as db:
        await db.execute(
            "UPDATE workflow_run_steps SET status='RETRYING', next_retry_at=?",
            (time.time() + 9999,),
        )
        await db.commit()
    assert await store.claim_next_approved_step(lease_owner="n", ttl_s=60) is None


# ----------------------------------------------------------------------
# Outcomes
# ----------------------------------------------------------------------


async def test_outcome_success_completes_run(tmp_path):
    store, wf_id = await _mk(tmp_path, executor_enabled=True)
    run, _ = await _approved_step(store, wf_id, executor_enabled=True)
    claimed = await store.claim_next_approved_step(lease_owner="n", ttl_s=60)

    done = await store.complete_step(
        claimed["step_uuid"], outcome_status="SUCCESS", actor="nexi-1"
    )
    assert done["status"] == "DONE"
    assert done["lease_owner"] is None  # lease released
    run_after = await store.get_run(run["id"])
    assert run_after["status"] == "COMPLETED"


async def test_outcome_failure_retries_then_fails_run(tmp_path):
    store, wf_id = await _mk(
        tmp_path, executor_enabled=True
    )
    run, _ = await _approved_step(store, wf_id, executor_enabled=True)

    # cycle 1: claim → fail → RETRYING with backoff
    c1 = await store.claim_next_approved_step(lease_owner="n", ttl_s=60)
    s1 = await store.complete_step(c1["step_uuid"], outcome_status="FAILURE", actor="n")
    assert s1["status"] == "RETRYING"
    assert s1["retry_count"] == 1
    assert s1["next_retry_at"] is not None and s1["next_retry_at"] > time.time()

    # not claimable before backoff elapses
    assert await store.claim_next_approved_step(lease_owner="n2", ttl_s=60) is None

    # force retry due, claim, fail again → retries exhausted (max 3) → FAILED
    import aiosqlite
    async with aiosqlite.connect(store._db) as db:
        for target in range(2, 4):  # retry_count 2, then 3
            await db.execute(
                "UPDATE workflow_run_steps SET next_retry_at=? WHERE step_uuid=?",
                (time.time() - 1, c1["step_uuid"]),
            )
            await db.commit()
            claimed = await store.claim_next_approved_step(lease_owner="n", ttl_s=60)
            assert claimed is not None
            s = await store.complete_step(
                claimed["step_uuid"], outcome_status="FAILURE", actor="n"
            )
            if target < 3:
                assert s["status"] == "RETRYING"
            else:
                assert s["status"] == "FAILED"

    run_after = await store.get_run(run["id"])
    assert run_after["status"] == "FAILED"


async def test_outcome_unknown_step_raises(tmp_path):
    store, _ = await _mk(tmp_path)
    with pytest.raises(_store_mod.WorkflowNotFound):
        await store.complete_step("nope", outcome_status="SUCCESS", actor="n")


# ----------------------------------------------------------------------
# Events trail includes lifecycle
# ----------------------------------------------------------------------


async def test_claim_and_outcome_append_events(tmp_path):
    store, wf_id = await _mk(tmp_path, executor_enabled=True)
    run, _ = await _approved_step(store, wf_id, executor_enabled=True)
    claimed = await store.claim_next_approved_step(lease_owner="n", ttl_s=60)
    uuid_ = claimed["step_uuid"]
    await store.complete_step(uuid_, outcome_status="SUCCESS", actor="n")
    kinds = [e["event_type"] for e in await store.step_events(uuid_)]
    assert kinds == ["RUN_CREATED", "APPROVED", "CLAIMED", "DONE"]
