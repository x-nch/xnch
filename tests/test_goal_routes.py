"""Goal HTTP endpoint tests: request/response shape + _serialize_goal row fixups."""
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from xnch.routes.goals import _serialize_goal, router as goal_router

_GOAL_ID = "11111111-1111-4111-8111-111111111111"


def _make_row(**overrides: object) -> dict[str, object]:
    """Build a raw DB row as GoalStore would return it (simulation_plan is a JSON string)."""
    row: dict[str, object] = {
        "goal_id": _GOAL_ID,
        "owner_actor_id": "actor-1",
        "objective": "deploy media service",
        "status": "PENDING",
        "progress": "",
        "steps_completed": 0,
        "max_steps": 10,
        "consecutive_failures": 0,
        "failure_threshold": 3,
        "last_step_outcome": None,
        "next_due_at": 123.0,
        "lease_owner": None,
        "lease_expires_at": None,
        "simulation_plan": json.dumps([{"action": "DEPLOY"}]),
        "created_at": 123.0,
        "updated_at": 123.0,
        "schema_version": "goal-v1",
    }
    row.update(overrides)
    return row


@pytest.fixture
def store() -> MagicMock:
    store = MagicMock()
    store.create_goal = AsyncMock(return_value="g-1")
    store.get_goal = AsyncMock(return_value=_make_row())
    store.list_goals = AsyncMock(return_value=[_make_row()])
    store.claim_next_goal = AsyncMock(return_value=_make_row(status="RUNNING"))
    store.update_goal = AsyncMock(return_value=_make_row())
    store.complete_step = AsyncMock(return_value=_make_row())
    return store


@pytest.fixture
def app(store: MagicMock) -> FastAPI:
    app = FastAPI()
    app.state.goal_store = store
    app.include_router(goal_router)
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_create_goal_returns_serialized_goal(client: httpx.AsyncClient, store: MagicMock):
    resp = await client.post(
        "/goals",
        json={
            "owner_actor_id": "actor-1",
            "objective": "deploy media service",
            "max_steps": 5,
            "failure_threshold": 2,
            "simulation_plan": [{"action": "DEPLOY"}],
        },
    )
    assert resp.status_code == 200
    store.create_goal.assert_awaited_once_with(
        owner_actor_id="actor-1",
        objective="deploy media service",
        max_steps=5,
        failure_threshold=2,
        simulation_plan=[{"action": "DEPLOY"}],
    )
    store.get_goal.assert_awaited_once_with("g-1")
    body = resp.json()
    assert body["simulation_plan"] == [{"action": "DEPLOY"}]
    assert "schema_version" not in body


async def test_list_goals_serializes_each_row(client: httpx.AsyncClient, store: MagicMock):
    resp = await client.get("/goals")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list) and len(body) == 1
    assert body[0]["simulation_plan"] == [{"action": "DEPLOY"}]
    assert "schema_version" not in body[0]
    store.list_goals.assert_awaited_once_with(None)


async def test_list_goals_passes_status_query(client: httpx.AsyncClient, store: MagicMock):
    resp = await client.get("/goals", params={"status": "ACTIVE"})
    assert resp.status_code == 200
    store.list_goals.assert_awaited_once_with("ACTIVE")


async def test_get_goal_returns_serialized(client: httpx.AsyncClient, store: MagicMock):
    resp = await client.get("/goals/g-123")
    assert resp.status_code == 200
    store.get_goal.assert_awaited_once_with("g-123")
    body = resp.json()
    assert body["simulation_plan"] == [{"action": "DEPLOY"}]
    assert "schema_version" not in body


async def test_get_goal_404_when_missing(client: httpx.AsyncClient, store: MagicMock):
    store.get_goal = AsyncMock(return_value=None)
    resp = await client.get("/goals/missing")
    assert resp.status_code == 404


async def test_claim_goal_returns_serialized(client: httpx.AsyncClient, store: MagicMock):
    resp = await client.post("/goals/claim", json={"lease_owner": "worker-1"})
    assert resp.status_code == 200
    store.claim_next_goal.assert_awaited_once_with("worker-1")
    body = resp.json()
    assert body["status"] == "RUNNING"
    assert body["simulation_plan"] == [{"action": "DEPLOY"}]
    assert "schema_version" not in body


async def test_claim_goal_null_when_none(client: httpx.AsyncClient, store: MagicMock):
    store.claim_next_goal = AsyncMock(return_value=None)
    resp = await client.post("/goals/claim", json={"lease_owner": "worker-1"})
    assert resp.status_code == 200
    assert resp.json() is None


async def test_update_goal_passes_status_and_progress(client: httpx.AsyncClient, store: MagicMock):
    resp = await client.post(
        "/goals/g-123/update", json={"status": "RUNNING", "progress": "step 1"}
    )
    assert resp.status_code == 200
    store.update_goal.assert_awaited_once_with("g-123", status="RUNNING", progress="step 1")
    assert "schema_version" not in resp.json()


async def test_update_goal_passes_none_through(client: httpx.AsyncClient, store: MagicMock):
    resp = await client.post("/goals/g-123/update", json={"status": None, "progress": None})
    assert resp.status_code == 200
    store.update_goal.assert_awaited_once_with("g-123", status=None, progress=None)


async def test_step_outcome_calls_complete_step(client: httpx.AsyncClient, store: MagicMock):
    resp = await client.post("/goals/g-123/step-outcome", json={"outcome_status": "SUCCESS"})
    assert resp.status_code == 200
    store.complete_step.assert_awaited_once_with("g-123", "SUCCESS")
    assert "schema_version" not in resp.json()


async def test_cancel_goal_sets_cancelled(client: httpx.AsyncClient, store: MagicMock):
    resp = await client.post("/goals/g-123/cancel")
    assert resp.status_code == 200
    store.update_goal.assert_awaited_once_with("g-123", status="CANCELLED")
    assert "schema_version" not in resp.json()


def test_serialize_goal_parses_simulation_plan_string():
    out = _serialize_goal(_make_row())
    assert out["simulation_plan"] == [{"action": "DEPLOY"}]
    assert "schema_version" not in out


def test_serialize_goal_empty_and_null_variants():
    assert _serialize_goal(_make_row(simulation_plan=""))["simulation_plan"] == []
    assert _serialize_goal(_make_row(simulation_plan="[]"))["simulation_plan"] == []
    assert _serialize_goal(_make_row(simulation_plan="not-json"))["simulation_plan"] == []
    assert _serialize_goal(_make_row(simulation_plan=[{"x": 1}]))["simulation_plan"] == [{"x": 1}]


def test_serialized_goal_validates_against_nexi_goal_model():
    from nexi.models.goal import Goal

    goal = Goal.model_validate(_serialize_goal(_make_row()))
    assert goal.simulation_plan == [{"action": "DEPLOY"}]
    assert goal.status.value == "PENDING"
