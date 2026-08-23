"""WorkflowStore — create/run/decide/expiry/idempotency + audit trail tests.

Environment note: the full xnch dependency tree (kuzu/langgraph/asyncpg/…)
is not installed in the minimal test venv, so memory/db.py and
memory/workflow_store.py are loaded via importlib (bypassing heavy package
__init__ chains). Both modules only need stdlib + aiosqlite + pydantic.
Production packaging is untouched.
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


_db = _load("xnch_wf_test_db", "memory/db.py")
_store = _load("xnch_wf_test_store", "memory/workflow_store.py")
init_db = _db.init_db
WorkflowStore = _store.WorkflowStore
ApprovalConflict = _store.ApprovalConflict
WorkflowNotFound = _store.WorkflowNotFound

STEPS = [
    {"id": "s1", "kind": "exec_tool", "summary": "Search highlights",
     "target": "web_search", "requires_approval": False},
    {"id": "s2", "kind": "write_file", "summary": "Draft weekly.md",
     "target": "reports/weekly.md", "requires_approval": True},
    {"id": "s3", "kind": "send_email", "summary": "Send to team@",
     "target": "team@x", "requires_approval": True},
]


async def _make_store(tmp_path):
    db_path = tmp_path / "wf.db"
    await init_db(db_path)
    store = WorkflowStore(db_path)
    wf_id = await store.create_workflow(
        owner_actor_id="operator",
        name="Weekly Digest",
        description=None,
        trigger={"kind": "manual"},
        steps=STEPS,
    )
    return store, wf_id


async def test_create_and_get_workflow_roundtrip(tmp_path):
    store, wf_id = await _make_store(tmp_path)
    row = await store.get_workflow(wf_id)
    assert row is not None and row["name"] == "Weekly Digest"
    assert json.loads(row["trigger_json"]) == {"kind": "manual"}
    assert len(json.loads(row["steps_json"])) == 3


async def test_update_and_list_and_delete(tmp_path):
    store, wf_id = await _make_store(tmp_path)
    updated = await store.update_workflow(wf_id, {"name": "Renamed"})
    assert updated["name"] == "Renamed"
    rows = await store.list_workflows()
    assert [r["id"] for r in rows] == [wf_id]
    assert await store.delete_workflow(wf_id) is True
    assert await store.get_workflow(wf_id) is None
    assert await store.delete_workflow(wf_id) is False


async def test_run_gating_enforced_including_elevated_kinds(tmp_path):
    """Elevated kinds are gated even when the stored flag says otherwise."""
    store, wf_id = await _make_store(tmp_path)
    run, created = await store.create_run(workflow_id=wf_id, actor="operator")
    assert created and run["status"] == "RUNNING"
    steps = json.loads(run["steps_json"])
    # s1 exec_tool was stored with requires_approval=False but is force-gated
    assert [s["status"] for s in steps] == ["AWAITING_APPROVAL"] * 3
    assert all(s["requires_approval"] for s in steps)
    approvals = await store.list_approvals(status="pending")
    assert len(approvals) == 3
    kinds = {json.loads(a["payload_json"])["kind"] for a in approvals}
    assert kinds == {"exec_tool", "write_file", "send_email"}
    risks = {a["risk_class"] for a in approvals}
    assert risks == {"low", "elevated"}  # write_file low; exec_tool/send_email elevated


async def test_non_elevated_kind_can_stay_ungated(tmp_path):
    """write_file with an explicit False flag remains ungated."""
    db_path = tmp_path / "ungated.db"
    await init_db(db_path)
    store = WorkflowStore(db_path)
    wf_id = await store.create_workflow(
        owner_actor_id="op", name="Local draft", description=None,
        trigger={"kind": "manual"},
        steps=[{"id": "w", "kind": "write_file", "summary": "scratch",
                "requires_approval": False}],
    )
    run, _ = await store.create_run(workflow_id=wf_id, actor="op")
    steps = json.loads(run["steps_json"])
    assert steps[0]["status"] == "DONE"
    assert await store.list_approvals(status="pending") == []


async def test_approve_all_completes_run(tmp_path):
    store, wf_id = await _make_store(tmp_path)
    run, _ = await store.create_run(workflow_id=wf_id, actor="op")
    for approval in await store.list_approvals(status="pending"):
        await store.decide_approval(approval["id"], decision="approve", actor="op")
    run_after = await store.get_run(run["id"])
    assert run_after["status"] == "COMPLETED"
    assert [s["status"] for s in json.loads(run_after["steps_json"])] == ["DONE"] * 3


async def test_reject_fails_the_run(tmp_path):
    store, wf_id = await _make_store(tmp_path)
    run, _ = await store.create_run(workflow_id=wf_id, actor="op")
    pending = await store.list_approvals(status="pending")
    await store.decide_approval(pending[0]["id"], decision="reject", actor="op")
    assert (await store.get_run(run["id"]))["status"] == "FAILED"


async def test_double_decide_conflicts(tmp_path):
    store, wf_id = await _make_store(tmp_path)
    await store.create_run(workflow_id=wf_id, actor="op")
    pending = await store.list_approvals(status="pending")
    await store.decide_approval(pending[0]["id"], decision="approve", actor="op")
    with pytest.raises(ApprovalConflict):
        await store.decide_approval(pending[0]["id"], decision="reject", actor="op")


async def test_idempotent_run_replays_existing(tmp_path):
    store, wf_id = await _make_store(tmp_path)
    r1, c1 = await store.create_run(workflow_id=wf_id, actor="op", idempotency_key="k1")
    r2, c2 = await store.create_run(workflow_id=wf_id, actor="op", idempotency_key="k1")
    assert c1 and not c2 and r1["id"] == r2["id"]
    assert len(await store.list_approvals(status="pending")) == 3


async def test_idempotent_decide_returns_same_row(tmp_path):
    store, wf_id = await _make_store(tmp_path)
    await store.create_run(workflow_id=wf_id, actor="op")
    pending = await store.list_approvals(status="pending")
    target = pending[0]["id"]
    d1 = await store.decide_approval(
        target, decision="approve", actor="op", idempotency_key="d1"
    )
    d2 = await store.decide_approval(
        target, decision="approve", actor="op", idempotency_key="d1"
    )
    assert d1["status"] == d2["status"] == "APPROVED"


async def test_lazy_expiry_flips_past_due_on_read(tmp_path):
    store, wf_id = await _make_store(tmp_path)
    await store.create_run(workflow_id=wf_id, actor="op", approval_ttl_s=-10.0)
    assert await store.list_approvals(status="pending") == []
    statuses = {a["status"] for a in await store.list_approvals(status=None)}
    assert "EXPIRED" in statuses


async def test_deciding_expired_approval_conflicts(tmp_path):
    store, wf_id = await _make_store(tmp_path)
    await store.create_run(workflow_id=wf_id, actor="op", approval_ttl_s=-10.0)
    rows = await store.list_approvals(status=None)  # triggers lazy expire
    with pytest.raises(ApprovalConflict):
        await store.decide_approval(rows[0]["id"], decision="approve", actor="op")


async def test_step_events_append_only_trail(tmp_path):
    store, wf_id = await _make_store(tmp_path)
    await store.create_run(workflow_id=wf_id, actor="op")
    pending = await store.list_approvals(status="pending")
    step_uuid = pending[0]["producer_id"]
    events0 = await store.step_events(step_uuid)
    assert [e["event_type"] for e in events0] == ["RUN_CREATED"]
    await store.decide_approval(
        pending[0]["id"], decision="approve", actor="op", note="looks good"
    )
    events1 = await store.step_events(step_uuid)
    assert [e["event_type"] for e in events1] == ["RUN_CREATED", "APPROVED"]
    assert json.loads(events1[-1]["snapshot_json"])["note"] == "looks good"


async def test_unknown_workflow_raises_not_found(tmp_path):
    store, _ = await _make_store(tmp_path)
    with pytest.raises(WorkflowNotFound):
        await store.create_run(workflow_id="nope", actor="op")


async def test_cancel_remaining_approvals_for_run(tmp_path):
    store, wf_id = await _make_store(tmp_path)
    run, _ = await store.create_run(workflow_id=wf_id, actor="op")
    cancelled = await store.cancel_approvals_for_run(run["id"], actor="admin")
    assert cancelled == 3
    statuses = {a["status"] for a in await store.list_approvals(status=None)}
    assert "CANCELLED" in statuses


def test_epoch_seconds_convention():
    now = time.time()
    assert 1_500_000_000 < now < 4_000_000_000
