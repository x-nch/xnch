"""xnch /memory/read + /memory/write experience reflection tests."""
import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from xnch.main import app as xnch_app
from xnch.memory.db import init_db
from xnch.memory.experience_store import ExperienceStore


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    return path


@pytest.fixture
def app_state(db_path):
    store = ExperienceStore(db_path)

    pg = MagicMock()
    pg.fetch_for_manifest = AsyncMock(return_value=[])
    pg.fetch_patterns_for_manifest = AsyncMock(return_value=[])

    state = MagicMock()
    state.pg_episodic = pg
    state.experience_store = store
    state.get_state_version = AsyncMock(return_value="v1.0.0")
    state.event_log = MagicMock()
    state.event_log.emit = MagicMock()
    return state


async def test_memory_read_returns_experiences(app_state):
    await app_state.experience_store.upsert_experience(
        context_signature="sha256:abc",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="operator",
        outcome="FAILURE",
        lesson="Rollback first, then stage",
        insight="Staging directly caused outage",
        verdict="MODIFY",
        applicability="EXECUTION|DEPLOY|SERVICE",
    )

    xnch_app.state = app_state
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=xnch_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/memory/read", json={
            "session_id": str(uuid4()),
            "actor_id": "act-1",
            "actor_role": "operator",
            "query": {
                "intent_class": "EXECUTION",
                "target_entity_class": "SERVICE",
                "lookback_window_days": 30,
                "max_episodes": 20,
                "max_patterns": 10,
            },
        })

    assert response.status_code == 200
    data = response.json()
    assert len(data["experiences"]) == 1
    assert data["experiences"][0]["lesson"] == "Rollback first, then stage"


async def test_memory_read_experiences_default_empty(app_state):
    xnch_app.state = app_state
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=xnch_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/memory/read", json={
            "session_id": str(uuid4()),
            "actor_id": "act-1",
            "actor_role": "operator",
            "query": {
                "intent_class": "EXECUTION",
                "target_entity_class": "SERVICE",
                "lookback_window_days": 30,
                "max_episodes": 20,
                "max_patterns": 10,
            },
        })

    assert response.status_code == 200
    assert response.json()["experiences"] == []


async def test_memory_write_experience_reflection(app_state):
    xnch_app.state = app_state
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=xnch_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/memory/write", json={
            "session_id": str(uuid4()),
            "actor_id": "act-1",
            "actor_role": "operator",
            "write_type": "EXPERIENCE_REFLECTION",
            "payload": {
                "context_signature": "sha256:abc",
                "intent_class": "EXECUTION",
                "action_type": "DEPLOY",
                "entity_class": "SERVICE",
                "actor_role": "operator",
                "outcome": "FAILURE",
                "lesson": "Rollback first, then stage",
                "insight": "Staging directly caused outage",
                "verdict": "MODIFY",
                "applicability": "EXECUTION|DEPLOY|SERVICE",
            },
        })

    assert response.status_code == 200

    rows = await app_state.experience_store.fetch_for_manifest(
        "EXECUTION", "SERVICE", "operator"
    )
    assert len(rows) == 1
    assert rows[0]["verdict"] == "MODIFY"


async def test_memory_write_experience_reflection_requires_payload_fields(app_state):
    xnch_app.state = app_state
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=xnch_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/memory/write", json={
            "session_id": str(uuid4()),
            "actor_id": "act-1",
            "actor_role": "operator",
            "write_type": "EXPERIENCE_REFLECTION",
            "payload": {"context_signature": "sha256:abc"},
        })

    assert response.status_code == 422
