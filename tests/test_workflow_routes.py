"""E2E: workflows + approvals over HTTP (FastAPI TestClient, real SQLite).

Builds minimal apps with ONLY the workflows/approvals routers — mirrors the
test_goal_routes.py pattern of exercising routers against app.state without
booting full main.py (whose dependency tree needs kuzu/langgraph/asyncpg).
Covers: create → run → queue → decide → run terminal → audit trail,
idempotent replay via Idempotency-Key, lazy expiry over HTTP, and the
Hybrid-B gateway-token gate.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

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


_db = _load("xnch_e2e_db", "memory/db.py")
_store_mod = _load("xnch_e2e_store", "memory/workflow_store.py")
_token = _load("xnch_e2e_token", "security/gateway_token.py")
init_db = _db.init_db
WorkflowStore = _store_mod.WorkflowStore
mint_gateway_token = _token.mint_gateway_token


def _build_app(store: WorkflowStore, secret: str = "") -> TestClient:
    # Load router module fresh per app (FastAPI routes register on the module's
    # router objects; two apps must not share registries).
    wf_routes = _load(f"xnch_e2e_routes_{id(store)}", "routes/workflows.py")
    app = FastAPI()
    app.state.workflow_store = store
    app.state.gateway_secret = secret
    app.include_router(wf_routes.router)
    app.include_router(wf_routes.approvals_router)
    return TestClient(app)


STEPS = [
    {"id": "s1", "kind": "exec_tool", "summary": "Search highlights",
     "target": "web_search", "requires_approval": True},
    {"id": "s2", "kind": "write_file", "summary": "Draft weekly.md",
     "target": "reports/weekly.md", "requires_approval": True},
    {"id": "s3", "kind": "send_email", "summary": "Send to team@",
     "target": "team@x", "requires_approval": False},
]


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "e2e.db"
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        store = loop.run_until_complete(_init_store(db_path))
    finally:
        loop.close()
    with _build_app(store) as client:
        yield client, store


async def _init_store(db_path):
    await init_db(db_path)
    return WorkflowStore(db_path)


def test_full_operator_journey_reject_path(env):
    c, store = env

    # 1. create workflow
    r = c.post(
        "/workflows",
        json={
            "name": "Weekly Digest",
            "description": "collect → draft → send",
            "trigger": {"kind": "manual"},
            "steps": STEPS,
            "owner_actor_id": "operator",
        },
    )
    assert r.status_code == 201, r.text
    wf = r.json()
    assert wf["name"] == "Weekly Digest" and len(wf["steps"]) == 3

    # 2. list + get
    assert any(w["id"] == wf["id"] for w in c.get("/workflows").json())
    assert c.get(f"/workflows/{wf['id']}").json()["description"] == "collect → draft → send"

    # 3. idempotent run (double-submit same key)
    r1 = c.post(f"/workflows/{wf['id']}/run", headers={"Idempotency-Key": "run-1"})
    assert r1.status_code == 201 and r1.json()["created"] is True
    r2 = c.post(f"/workflows/{wf['id']}/run", headers={"Idempotency-Key": "run-1"})
    assert r2.status_code == 201 and r2.json()["created"] is False
    assert r1.json()["id"] == r2.json()["id"]
    run_id = r1.json()["id"]

    # 4. unified queue — exactly 2 pending (non-gated step auto-done)
    queue = c.get("/approvals?status=pending").json()
    assert len(queue) == 2
    assert {a["payload"]["kind"] for a in queue} == {"exec_tool", "write_file"}
    assert all(a["producer_type"] == "workflow_step" for a in queue)

    # 5. reject one → run FAILED; approve other; re-decide → 409
    d1 = c.post(
        f"/approvals/{queue[0]['id']}/decide",
        json={"decision": "reject", "note": "wrong quarter"},
        headers={"X-Actor-Id": "operator-7"},
    )
    assert d1.status_code == 200
    body = d1.json()
    assert body["status"] == "REJECTED" and body["decided_by"] == "operator-7"

    remaining = c.get("/approvals?status=pending").json()
    assert len(remaining) == 1
    assert c.post(
        f"/approvals/{remaining[0]['id']}/decide", json={"decision": "approve"}
    ).status_code == 200

    replay = c.post(f"/approvals/{queue[0]['id']}/decide", json={"decision": "approve"})
    assert replay.status_code == 409

    runs = c.get("/workflows/runs", params={"workflow_id": wf["id"]}).json()
    assert runs[0]["id"] == run_id and runs[0]["status"] == "FAILED"

    # 6. append-only audit trail for the rejected step
    events = c.get(f"/approvals/{queue[0]['id']}/events").json()
    assert [e["event_type"] for e in events] == ["RUN_CREATED", "REJECTED"]
    assert json.loads(events[-1]["snapshot_json"])["note"] == "wrong quarter"


def test_approve_all_completes_run(env):
    c, store = env
    wf = c.post(
        "/workflows", json={"name": "Research", "steps": STEPS, "owner_actor_id": "op"}
    ).json()
    assert c.post(f"/workflows/{wf['id']}/run").status_code == 201
    for approval in c.get("/approvals?status=pending").json():
        assert (
            c.post(f"/approvals/{approval['id']}/decide", json={"decision": "approve"}).status_code
            == 200
        )
    runs = c.get("/workflows/runs", params={"workflow_id": wf["id"]}).json()
    assert runs[0]["status"] == "COMPLETED"
    assert [s["status"] for s in runs[0]["steps"]] == ["DONE"] * 3


def test_patch_and_delete_workflow(env):
    c, store = env
    wf = c.post(
        "/workflows", json={"name": "Old", "steps": STEPS, "owner_actor_id": "op"}
    ).json()
    patched = c.patch(f"/workflows/{wf['id']}", json={"name": "New"})
    assert patched.status_code == 200 and patched.json()["name"] == "New"
    assert c.delete(f"/workflows/{wf['id']}").status_code == 204
    assert c.get(f"/workflows/{wf['id']}").status_code == 404
    assert c.delete(f"/workflows/{wf['id']}").status_code == 404


def test_lazy_expiry_over_http(tmp_path):
    import asyncio

    async def mk():
        db = tmp_path / "exp.db"
        await init_db(db)
        store = WorkflowStore(db)
        wf_id = await store.create_workflow(
            owner_actor_id="op", name="TTL", description=None,
            trigger={"kind": "manual"},
            steps=[{"id": "a", "kind": "write_file", "summary": "x",
                    "requires_approval": True}],
        )
        await store.create_run(workflow_id=wf_id, actor="op", approval_ttl_s=-5)
        return store

    store = asyncio.new_event_loop().run_until_complete(mk())
    with _build_app(store) as c:
        assert c.get("/approvals?status=pending").json() == []  # expired inline
        decided = c.get("/approvals", params={"status": "EXPIRED"}).json()
        assert {a["status"] for a in decided} == {"EXPIRED"}


def test_gateway_token_gate_blocks_unsigned_writes(tmp_path):
    import asyncio

    async def mk():
        db = tmp_path / "gate.db"
        await init_db(db)
        return WorkflowStore(db)

    store = asyncio.new_event_loop().run_until_complete(mk())
    secret = "unit-test-secret"
    with _build_app(store, secret=secret) as c:
        body = {"name": "W", "steps": STEPS, "owner_actor_id": "op"}

        # unsigned write → 401
        assert c.post("/workflows", json=body).status_code == 401
        # malformed token → 401
        bad = c.post("/workflows", json=body,
                     headers={"X-Gateway-Token": "not-a-token"})
        assert bad.status_code == 401
        # expired token → 401
        expired = mint_gateway_token(secret, ttl_s=-10)
        assert c.post("/workflows", json=body,
                      headers={"X-Gateway-Token": expired}).status_code == 401
        # valid token → 201
        token = mint_gateway_token(secret, ttl_s=60)
        ok = c.post("/workflows", json=body, headers={"X-Gateway-Token": token})
        assert ok.status_code == 201, ok.text
        # service key accepted (nexi identity, Phase 2)
        svc = c.post("/workflows", json=body, headers={"X-Service-Key": secret})
        assert svc.status_code == 201
        # reads stay ungated
        assert c.get("/workflows").status_code == 200

        # decide is also gated
        wf = ok.json()
        c.post(f"/workflows/{wf['id']}/run", headers={"X-Gateway-Token": mint_gateway_token(secret)})
        approvals = c.get("/approvals?status=pending").json()
        approval = approvals[0]
        assert c.post(f"/approvals/{approval['id']}/decide",
                      json={"decision": "approve"}).status_code == 401
        good = c.post(
            f"/approvals/{approval['id']}/decide",
            json={"decision": "approve"},
            headers={"X-Gateway-Token": mint_gateway_token(secret)},
        )
        assert good.status_code == 200


def test_epoch_seconds_sanity():
    now = time.time()
    assert 1_500_000_000 < now < 4_000_000_000
