"""E2E for P2 claim/outcome endpoints (service-key gated).

Fails until /workflows/steps/claim and /workflows/steps/{uuid}/outcome exist.
"""
from __future__ import annotations

import importlib.util
import sys
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


_db = _load("xnch_p2r_db", "memory/db.py")
_store_mod = _load("xnch_p2r_store", "memory/workflow_store.py")
init_db = _db.init_db
WorkflowStore = _store_mod.WorkflowStore

STEPS = [
    {"id": "s1", "kind": "exec_tool", "summary": "Search",
     "target": "web_search", "requires_approval": True},
]


def _build(store, secret=""):
    wf_routes = _load(f"xnch_p2r_routes_{id(store)}", "routes/workflows.py")
    app = FastAPI()
    app.state.workflow_store = store
    app.state.gateway_secret = secret
    app.include_router(wf_routes.router)
    app.include_router(wf_routes.approvals_router)
    return TestClient(app)


def _mk_store(tmp_path, executor_enabled=True):
    import asyncio

    async def mk():
        db = tmp_path / f"p2r_{id(tmp_path)}.db"
        await init_db(db)
        return WorkflowStore(db, executor_enabled=executor_enabled)

    return asyncio.new_event_loop().run_until_complete(mk())


def test_claim_and_outcome_over_http(tmp_path):
    store = _mk_store(tmp_path)
    c = _build(store)

    wf = c.post(
        "/workflows", json={"name": "W", "steps": STEPS, "owner_actor_id": "op"}
    ).json()
    run_id = c.post(f"/workflows/{wf['id']}/run").json()["id"]

    # approve → APPROVED (executor-enabled store), not terminal
    (approval,) = c.get("/approvals?status=pending").json()
    assert (
        c.post(
            f"/approvals/{approval['id']}/decide",
            json={"decision": "approve"},
            headers={"X-Actor-Role": "admin"},
        ).status_code
        == 200
    )
    rows = store.get_run_step_rows_sync_hack(run_id) if hasattr(store, "get_run_step_rows_sync_hack") else None
    assert rows is None  # no sync hack in production API

    # empty claim before approval? — approval already given above; claim now
    claimed = c.post("/workflows/steps/claim", json={"lease_owner": "nexi-1"})
    assert claimed.status_code == 200, claimed.text
    step = claimed.json()
    assert step["status"] == "CLAIMED" and step["lease_owner"] == "nexi-1"

    # outcome SUCCESS → DONE; run completes
    out = c.post(
        f"/workflows/steps/{step['step_uuid']}/outcome",
        json={"outcome_status": "SUCCESS"},
    )
    assert out.status_code == 200
    assert out.json()["status"] == "DONE"

    runs = c.get("/workflows/runs", params={"workflow_id": wf["id"]}).json()
    assert runs[0]["status"] == "COMPLETED"


def test_claim_returns_204_when_nothing_to_claim(tmp_path):
    store = _mk_store(tmp_path)
    c = _build(store)
    resp = c.post("/workflows/steps/claim", json={"lease_owner": "nexi-1"})
    assert resp.status_code == 204


def test_outcome_unknown_step_404(tmp_path):
    store = _mk_store(tmp_path)
    c = _build(store)
    resp = c.post(
        "/workflows/steps/nope/outcome", json={"outcome_status": "SUCCESS"}
    )
    assert resp.status_code == 404


def test_service_key_gate_on_executor_endpoints(tmp_path):
    store = _mk_store(tmp_path)
    secret = "svc-secret"
    c = _build(store, secret=secret)

    wf = c.post(
        "/workflows",
        json={"name": "W", "steps": STEPS, "owner_actor_id": "op"},
        headers={"X-Service-Key": secret},
    ).json()
    c.post(
        f"/workflows/{wf['id']}/run",
        json=None,
        headers={"X-Service-Key": secret},
    )
    (approval,) = c.get("/approvals?status=pending").json()
    assert (
        c.post(
            f"/approvals/{approval['id']}/decide",
            json={"decision": "approve"},
            headers={"X-Service-Key": secret, "X-Actor-Role": "admin"},
        ).status_code
        == 200
    )

    # unsigned claim → 401
    assert (
        c.post("/workflows/steps/claim", json={"lease_owner": "n"}).status_code == 401
    )
    # service key → 200
    ok = c.post(
        "/workflows/steps/claim",
        json={"lease_owner": "n"},
        headers={"X-Service-Key": secret},
    )
    assert ok.status_code == 200
