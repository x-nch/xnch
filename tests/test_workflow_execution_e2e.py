"""Phase C end-to-end execution — one minimal workflow (2 steps, 1 gated),
real store lifecycle in executor mode: gate halt → decision → nexi-side
claim/execute/outcome → audit trail.

Mirrors how nexi/workflow/executor.py drives the store over HTTP; execute_fn
is simulated inline so the suite needs no pipeline dependency tree.
"""
from __future__ import annotations

import importlib.util
import json
import sys
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


_db = _load("xnch_c_db", "memory/db.py")
_store_mod = _load("xnch_c_store", "memory/workflow_store.py")
init_db = _db.init_db
WorkflowStore = _store_mod.WorkflowStore

# The brief's minimal case: two steps, one requiresApproval. The ungated
# write_file stays auto-done (v1 semantic); exec_tool would be force-gated
# server-side even if the flag were false (Phase B enforcement).
STEPS = [
    {"id": "s1", "kind": "write_file", "summary": "Draft report",
     "target": "reports/draft.md", "requires_approval": False},
    {"id": "s2", "kind": "exec_tool", "summary": "Publish via tool",
     "target": "publish", "args": {"channel": "blog"},
     "requires_approval": True},
]


async def _mk(tmp_path):
    db_path = tmp_path / "e2e_exec.db"
    await init_db(db_path)
    store = WorkflowStore(db_path, executor_enabled=True)
    wf_id = await store.create_workflow(
        owner_actor_id="op", name="Draft then publish",
        description=None, trigger={"kind": "manual"}, steps=STEPS,
    )
    return store, wf_id


async def test_gate_halts_execution_until_decided(tmp_path):
    """While AWAITING_APPROVAL, the executor can claim nothing."""
    store, wf_id = await _mk(tmp_path)
    run, _ = await store.create_run(workflow_id=wf_id, actor="op")

    # halt: nothing claimable pre-decision
    assert await store.claim_next_approved_step(lease_owner="nexi", ttl_s=60) is None

    rows = {r["status"] for r in await store.get_run_step_rows(run["id"])}
    assert rows == {"DONE", "AWAITING_APPROVAL"}

    pendings = await store.list_approvals(status="pending")
    assert len(pendings) == 1
    assert pendings[0]["producer_type"] == "workflow_step"
    payload = json.loads(pendings[0]["payload_json"])
    assert payload["kind"] == "exec_tool"
    assert payload["target"] == "publish"
    assert (await store.get_run(run["id"]))["status"] == "RUNNING"


async def test_two_steps_one_gated_full_lifecycle(tmp_path):
    store, wf_id = await _mk(tmp_path)
    run, _ = await store.create_run(workflow_id=wf_id, actor="op")

    (approval,) = await store.list_approvals(status="pending")
    await store.decide_approval(approval["id"], decision="approve", actor="operator-1")

    # approve ⇒ APPROVED (executor mode), not DONE
    claimed = await store.claim_next_approved_step(lease_owner="nexi-wf", ttl_s=120)
    assert claimed is not None
    assert claimed["status"] == "CLAIMED"
    # payload round-trips to the executor for the pipeline pass
    assert claimed["kind"] == "exec_tool"
    claimed_payload = json.loads(claimed["payload_json"] or "{}")
    assert claimed_payload["target"] == "publish"
    assert claimed_payload.get("args") == {"channel": "blog"}

    done = await store.complete_step(
        claimed["step_uuid"], outcome_status="SUCCESS", actor="nexi-wf"
    )
    assert done["status"] == "DONE"

    run_after = await store.get_run(run["id"])
    assert run_after["status"] == "COMPLETED"

    # audit trail for the gated step — durable beyond final status
    events = await store.step_events(claimed["step_uuid"])
    assert [e["event_type"] for e in events] == [
        "RUN_CREATED", "APPROVED", "CLAIMED", "DONE",
    ]
    assert events[1]["actor"] == "operator-1"
    assert events[-1]["actor"] == "nexi-wf"


async def test_reject_terminates_run_in_executor_mode(tmp_path):
    store, wf_id = await _mk(tmp_path)
    run, _ = await store.create_run(workflow_id=wf_id, actor="op")

    (approval,) = await store.list_approvals(status="pending")
    await store.decide_approval(
        approval["id"], decision="reject", actor="operator-2", note="not ready"
    )

    rows = await store.get_run_step_rows(run["id"])
    by_status = {r["status"] for r in rows}
    assert "REJECTED" in by_status
    assert (await store.get_run(run["id"]))["status"] == "FAILED"

    # terminated: nothing left to execute, nothing re-decidable
    assert await store.claim_next_approved_step(lease_owner="nexi", ttl_s=60) is None
    with pytest.raises(_store_mod.ApprovalConflict):
        await store.decide_approval(approval["id"], decision="approve", actor="op")


async def test_timeout_expiry_releases_the_gate(tmp_path):
    """TTL lapse expires the approval and unblocks the queue (lazy sweep)."""
    store, wf_id = await _mk(tmp_path)
    run, _ = await store.create_run(
        workflow_id=wf_id, actor="op", approval_ttl_s=-1
    )
    assert await store.list_approvals(status="pending") == []
    decided = await store.list_approvals(status=None)
    assert {a["status"] for a in decided} == {"EXPIRED"}
    rows = await store.get_run_step_rows(run["id"])
    assert any(r["status"] == "EXPIRED" for r in rows)
