"""E2E: goal-step auto-dispatch loop — cron filing, gate, spawn, back-pressure.

Real SQLite + real stores (GoalStore/WorkflowStore/AgentRunStore), minimal
FastAPI apps per the established loader pattern. Covers:
- init_db migration adding agent_runs.approval_id to legacy tables
- run_due_dispatch guards (missing goal / exhausted / already-pending) and the
  happy path (approval filed from simulation_plan[steps_completed])
- approve-side spawn creating a linked agent_run
- outcome back-pressure: DONE -> goal SUCCESS step, FAILED -> FAILURE step
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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


_db = _load("xgd_db", "memory/db.py")
_goals = _load("xgd_goals", "memory/goal_store.py")
_wf = _load("xgd_wf", "memory/workflow_store.py")
_ar = _load("xgd_ar", "memory/agent_run_store.py")
_gd = _load("xgd_job", "jobs/goal_dispatch.py")
init_db = _db.init_db
GoalStore, WorkflowStore, AgentRunStore = _goals.GoalStore, _wf.WorkflowStore, _ar.AgentRunStore
run_due_dispatch = _gd.run_due_dispatch
apply_bp = _gd.apply_outcome_backpressure
spawn_agent_run_for_approval = _gd.spawn_agent_run_for_approval

GOAL_ID = "11111111-1111-4111-8111-111111111111"
PLAN = [
    {"day": 1, "action": "shortlist companies", "output": "target-list.md"},
    {"day": 2, "action": "resume rewrite", "output": "resume.md"},
]


@pytest.fixture()
async def env(tmp_path: Path):
    db = tmp_path / "xnch.db"
    await init_db(db)
    return {
        "db": db,
        "goals": GoalStore(db),
        "wf": WorkflowStore(db),
        "agents": AgentRunStore(db),
    }


async def _seed_goal(env) -> dict:
    await env["goals"].create_goal(
        owner_actor_id="ck-san",
        objective="Get a new job within 15 days (deadline 2026-09-07)",
        max_steps=15,
        simulation_plan=PLAN,
    )
    # create_goal generates its own id — overwrite objective row id via update? No:
    # fetch the single goal and use its real id.
    rows = await env["goals"].list_goals()
    assert len(rows) == 1
    return rows[0]


def _mint(secret):
    tok = _load(f"xgd_tok_{id(secret)}", "security/gateway_token.py")
    return {"X-Gateway-Token": tok.mint_gateway_token(secret)}


async def test_migration_adds_approval_id_to_legacy_table(tmp_path: Path) -> None:
    import aiosqlite

    db = tmp_path / "legacy.db"
    async with aiosqlite.connect(db) as d:
        await d.execute(
            "CREATE TABLE agent_runs (id TEXT PRIMARY KEY, status TEXT,"
            " prompt TEXT, workspace TEXT, runner_id TEXT, lease_expires_at REAL,"
            " exit_code INTEGER, output_path TEXT, error TEXT,"
            " created_at REAL, updated_at REAL)"
        )
        await d.commit()
    await init_db(db)
    async with aiosqlite.connect(db) as d:
        cur = await d.execute("PRAGMA table_info(agent_runs)")
        cols = {row[1] for row in await cur.fetchall()}
    assert "approval_id" in cols


async def test_dispatch_happy_path_files_approval_from_plan(env) -> None:
    goal = await _seed_goal(env)
    out = await run_due_dispatch(
        goal_store=env["goals"], workflow_store=env["wf"],
        agent_run_store=env["agents"], goal_id=goal["goal_id"],
    )
    assert "approval_id" in out and out["step_index"] == 0
    pending = await env["wf"].pending_goal_approval(goal["goal_id"])
    assert pending is not None and pending["status"] == "AWAITING_APPROVAL"
    payload = json.loads(pending["payload_json"])
    assert payload["step_index"] == 0
    assert "shortlist companies" in payload["prompt"]
    assert "target-list.md" in payload["prompt"]


async def test_dispatch_skips_when_already_pending(env) -> None:
    goal = await _seed_goal(env)
    kw = dict(goal_store=env["goals"], workflow_store=env["wf"], agent_run_store=env["agents"], goal_id=goal["goal_id"])
    first = await run_due_dispatch(**kw)
    second = await run_due_dispatch(**kw)
    assert "approval_id" in first and second == {"skipped": "approval already pending"}


async def test_dispatch_skips_unknown_goal_and_exhausted(env) -> None:
    kw = dict(goal_store=env["goals"], workflow_store=env["wf"], agent_run_store=env["agents"], goal_id="nope")
    assert "not found" in (await run_due_dispatch(**kw))["skipped"]

    goal = await _seed_goal(env)
    for _ in range(2):  # plan has 2 entries; burn both steps via gate+spawn+outcome
        r = await run_due_dispatch(**{**kw, "goal_id": goal["goal_id"]})
        ap = await env["wf"].pending_goal_approval(goal["goal_id"])
        await env["wf"].decide_approval(ap["id"], decision="approve", actor="t")
        await spawn_agent_run_for_approval(agent_run_store=env["agents"], approval=ap)
        run_row = (await env["agents"].list_runs())[0]
        await env["agents"].complete_run(run_row["id"], outcome_status="DONE", exit_code=0)
        await apply_bp(agent_run_store=env["agents"], workflow_store=env["wf"],
                       goal_store=env["goals"], run_row=(await env["agents"].get_run(run_row["id"])))
    final = await run_due_dispatch(**{**kw, "goal_id": goal["goal_id"]})
    assert final == {"skipped": "plan shorter than steps_completed"}


async def test_full_loop_approve_spawn_outcome_advances_goal(env) -> None:
    """The whole v1 promise: gate -> dispatch -> outcome -> goal advances."""
    goal = await _seed_goal(env)
    gid = goal["goal_id"]
    await run_due_dispatch(goal_store=env["goals"], workflow_store=env["wf"],
                           agent_run_store=env["agents"], goal_id=gid)
    ap = await env["wf"].pending_goal_approval(gid)

    decided = await env["wf"].decide_approval(ap["id"], decision="approve", actor="ck-san")
    assert decided["status"] == "APPROVED"

    # Hook is a route-level concern; store-level test invokes it explicitly.
    await spawn_agent_run_for_approval(agent_run_store=env["agents"], approval=ap)

    runs = await env["agents"].list_runs()
    assert len(runs) == 1 and runs[0]["approval_id"] == ap["id"]
    assert "shortlist companies" in runs[0]["prompt"]

    claimed = await env["agents"].claim_next("test-runner", ttl_s=600)
    assert claimed is not None and claimed["id"] == runs[0]["id"]

    after_done = await env["agents"].complete_run(runs[0]["id"], outcome_status="DONE", exit_code=0)
    assert after_done["status"] == "DONE"
    await apply_bp(agent_run_store=env["agents"], workflow_store=env["wf"],
                   goal_store=env["goals"], run_row=after_done)

    g = await env["goals"].get_goal(gid)
    assert g["steps_completed"] == 1
    assert g["last_step_outcome"] == "SUCCESS"


async def test_route_level_gate_and_backpressure(env) -> None:
    """HTTP path: decide via approvals router spawns run; agents outcome advances goal."""
    routes = _load(f"xgd_routes_{uuid4().hex}", "routes/workflows.py")
    aroutes = _load(f"xgd_aroutes_{uuid4().hex}", "routes/agents.py")
    app = FastAPI()
    app.state.workflow_store = env["wf"]
    app.state.agent_run_store = env["agents"]
    app.state.goal_store = env["goals"]
    app.state.gateway_secret = ""  # open gate for test brevity; 401s covered elsewhere
    tc = TestClient(app)
    tc.app.include_router(routes.approvals_router)
    tc.app.include_router(aroutes.router)

    goal = await _seed_goal(env)
    gid = goal["goal_id"]
    await run_due_dispatch(goal_store=env["goals"], workflow_store=env["wf"],
                           agent_run_store=env["agents"], goal_id=gid)
    ap = await env["wf"].pending_goal_approval(gid)

    ok = tc.post(f"/approvals/{ap['id']}/decide", json={"decision": "approve"})
    assert ok.status_code == 200
    runs = await env["agents"].list_runs()
    assert runs and runs[0]["approval_id"] == ap["id"]

    claim = tc.post("/agents/dispatch/next",
                    json={"runner_id": "test-runner", "ttl_s": 600})
    assert claim.status_code == 200 and claim.json()["id"] == runs[0]["id"]

    done = tc.post(
        f"/agents/runs/{runs[0]['id']}/outcome",
        json={"outcome_status": "DONE", "exit_code": 0},
    )
    assert done.status_code == 200
    g = await env["goals"].get_goal(gid)
    assert g["steps_completed"] == 1 and g["last_step_outcome"] == "SUCCESS"

    # Reject leg: next day's gate rejected -> failure recorded, nothing dispatches.
    await run_due_dispatch(goal_store=env["goals"], workflow_store=env["wf"],
                           agent_run_store=env["agents"], goal_id=gid)
    ap2 = await env["wf"].pending_goal_approval(gid)
    rej = tc.post(f"/approvals/{ap2['id']}/decide", json={"decision": "reject"})
    assert rej.status_code == 200
    assert len(await env["agents"].list_runs()) == 1  # still only the approved run
    g2 = await env["goals"].get_goal(gid)
    assert g2["last_step_outcome"] == "FAILURE"
    assert g2["consecutive_failures"] == 1
