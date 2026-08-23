"""Phase D — the integration question this whole feature had to answer:

Do workflow_step approvals surface in the SAME pending queue as live
goal_step approvals, decidable through the SAME decide path, audited in
the SAME step_events log?

One store, one table, one list, one decider — both producers side by side.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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


_db = _load("xnch_d_db", "memory/db.py")
_store_mod = _load("xnch_d_store", "memory/workflow_store.py")
init_db = _db.init_db
WorkflowStore = _store_mod.WorkflowStore


async def test_workflow_and_goal_approvals_share_one_queue(tmp_path):
    await init_db(tmp_path / "unified.db")
    store = WorkflowStore(tmp_path / "unified.db", executor_enabled=True)

    # producer 1: a workflow run's gated step (this feature)
    wf_id = await store.create_workflow(
        owner_actor_id="op", name="WF", description=None,
        trigger={"kind": "manual"},
        steps=[{"id": "s1", "kind": "exec_tool", "summary": "Tool step",
                "requires_approval": True}],
    )
    run, _ = await store.create_run(workflow_id=wf_id, actor="op")

    # producer 2: goal-dispatch's approval (live system)
    goal_approval = await store.create_goal_approval(
        goal_id="goal-123",
        payload={"summary": "Ship weekly digest", "target": "email"},
    )

    # THE assertion: one unfiltered pending queue holds both producer types.
    queue = await store.list_approvals(status="pending")
    types = {a["producer_type"] for a in queue}
    assert types == {"workflow_step", "goal_step"}
    ids = {a["id"] for a in queue}
    assert len(queue) == 2 and len(ids) == 2

    # both decided through the SAME decide_approval state machine
    for approval in queue:
        row = await store.decide_approval(
            approval["id"], decision="approve", actor="operator-9"
        )
        assert row["status"] == "APPROVED"

    # same audit log serves both producers (keyed by producer_id/step_uuid)
    wf_step_uuid = json.loads(
        next(a for a in queue if a["producer_type"] == "workflow_step")["payload_json"]
    )["run_id"]
    rows = await store.get_run_step_rows(run["id"])
    wf_events = await store.step_events(rows[0]["step_uuid"])
    assert [e["event_type"] for e in wf_events] == ["RUN_CREATED", "APPROVED"]
    goal_events = await store.step_events(goal_approval["producer_id"])
    assert [e["event_type"] for e in goal_events] == ["CREATED", "APPROVED"]
    created_snapshot = json.loads(goal_events[0]["snapshot_json"])
    assert created_snapshot["summary"] == "Ship weekly digest"
    assert wf_step_uuid  # payload linkage intact for UI drill-down

    # filtered views still work for per-producer dashboards
    only_wf = await store.list_approvals(status=None, producer_type="workflow_step")
    assert {a["producer_type"] for a in only_wf} == {"workflow_step"}


async def test_decisions_from_both_producers_land_in_one_chronology(tmp_path):
    """Interleaved decisions stay ordered in one audit stream per step."""
    await init_db(tmp_path / "chrono.db")
    store = WorkflowStore(tmp_path / "chrono.db", executor_enabled=True)

    wf_id = await store.create_workflow(
        owner_actor_id="op", name="WF", description=None,
        trigger={"kind": "manual"},
        steps=[{"id": "s1", "kind": "send_email", "summary": "Send",
                "requires_approval": True}],
    )
    await store.create_run(workflow_id=wf_id, actor="op")
    goal_approval = await store.create_goal_approval(
        goal_id="goal-777", payload={"summary": "Goal side"}
    )

    queue = await store.list_approvals(status="pending")
    by_type = {a["producer_type"]: a["id"] for a in queue}

    # interleave: approve goal first, then reject workflow step
    await store.decide_approval(by_type["goal_step"], decision="approve", actor="a1")
    await store.decide_approval(
        by_type["workflow_step"], decision="reject", actor="a2", note="no send"
    )

    goal_events = await store.step_events(goal_approval["producer_id"])
    assert [e["event_type"] for e in goal_events] == ["CREATED", "APPROVED"]
    assert goal_events[-1]["actor"] == "a1"

    rows = await store.get_run_step_rows(
        (await store.list_runs())[0]["id"]
    )
    wf_events = await store.step_events(rows[0]["step_uuid"])
    assert [e["event_type"] for e in wf_events] == ["RUN_CREATED", "REJECTED"]
    assert wf_events[-1]["actor"] == "a2"
    snapshot = json.loads(wf_events[-1]["snapshot_json"])
    assert snapshot["note"] == "no send"
