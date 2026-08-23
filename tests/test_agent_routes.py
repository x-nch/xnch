"""E2E: agent dispatch queue over HTTP (FastAPI TestClient, real SQLite).

Mirrors test_workflow_routes.py: minimal app with ONLY the agents router,
fresh module load per app, real tmp SQLite via init_db + AgentRunStore.
Covers: dispatch 201/401, FIFO claim then 204 empty, outcome happy/404/409,
open read list.
"""
from __future__ import annotations

import importlib.util
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


_db = _load("xnch_agents_db", "memory/db.py")
_store_mod = _load("xnch_agents_store", "memory/agent_run_store.py")
_token = _load("xnch_agents_token", "security/gateway_token.py")
init_db = _db.init_db
AgentRunStore = _store_mod.AgentRunStore
mint_gateway_token = _token.mint_gateway_token

SECRET = "test-secret"


def _token_header() -> dict[str, str]:
    return {"X-Gateway-Token": mint_gateway_token(SECRET)}


def _build_app(store: AgentRunStore, secret: str = SECRET) -> TestClient:
    routes = _load(f"xnch_agents_routes_{id(store)}", "routes/agents.py")
    app = FastAPI()
    app.state.agent_run_store = store
    app.state.gateway_secret = secret
    app.include_router(routes.router)
    return TestClient(app)


@pytest.fixture()
async def client(tmp_path: Path):
    db = tmp_path / "xnch.db"
    await init_db(db)
    store = AgentRunStore(db)
    yield _build_app(store), store


def test_dispatch_requires_token(client) -> None:
    tc, _ = client
    r = tc.post("/agents/dispatch", json={"prompt": "x"})
    assert r.status_code == 401


def test_dispatch_creates_queued_run(client) -> None:
    tc, _ = client
    r = tc.post(
        "/agents/dispatch",
        json={"prompt": "create hello.txt"},
        headers=_token_header(),
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "QUEUED"
    assert body["workspace"].startswith("~/xnch-agents/")


def test_claim_fifo_then_empty_204(client) -> None:
    tc, _ = client
    for i in range(2):
        tc.post("/agents/dispatch", json={"prompt": f"task-{i}"}, headers=_token_header())
    h = _token_header()

    first = tc.post("/agents/dispatch/next", json={"runner_id": "mac", "ttl_s": 60}, headers=h)
    assert first.status_code == 200
    assert first.json()["prompt"] == "task-0"
    assert first.json()["status"] == "RUNNING"

    second = tc.post("/agents/dispatch/next", json={"runner_id": "mac", "ttl_s": 60}, headers=h)
    assert second.status_code == 200
    assert second.json()["prompt"] == "task-1"

    empty = tc.post("/agents/dispatch/next", json={"runner_id": "mac", "ttl_s": 60}, headers=h)
    assert empty.status_code == 204


def test_outcome_happy_404_and_409(client) -> None:
    tc, _ = client
    run_id = tc.post(
        "/agents/dispatch", json={"prompt": "p"}, headers=_token_header()
    ).json()["id"]
    h = _token_header()

    # 409: cannot complete a QUEUED run.
    queued = tc.post(f"/agents/runs/{run_id}/outcome", json={"outcome_status": "DONE"}, headers=h)
    assert queued.status_code == 409

    claim = tc.post("/agents/dispatch/next", json={"runner_id": "mac", "ttl_s": 60}, headers=h)
    assert claim.status_code == 200

    ok = tc.post(
        f"/agents/runs/{run_id}/outcome",
        json={"outcome_status": "DONE", "exit_code": 0, "result_text": "the answer"},
        headers=h,
    )
    assert ok.status_code == 200
    assert ok.json()["status"] == "DONE"

    detail = tc.get(f"/agents/runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["result_text"] == "the answer"

    missing_detail = tc.get("/agents/runs/nope")
    assert missing_detail.status_code == 404

    missing = tc.post("/agents/runs/nope/outcome", json={"outcome_status": "DONE"}, headers=h)
    assert missing.status_code == 404


def test_list_runs_open_read(client) -> None:
    tc, _ = client
    tc.post("/agents/dispatch", json={"prompt": "visible"}, headers=_token_header())
    r = tc.get("/agents/runs")  # no token — reads are open
    assert r.status_code == 200
    assert [x["prompt"] for x in r.json()] == ["visible"]

    filtered = tc.get("/agents/runs?status=RUNNING")
    assert filtered.json() == []
