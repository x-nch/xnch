"""LangGraph decision pipeline wiring tests."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from xnch.main import app as xnch_app


@pytest.fixture
def langgraph_app_state(mock_app_state):
    """App state with a mocked decision runner."""
    runner = MagicMock()
    runner.run = AsyncMock(return_value={
        "status": "EXECUTING",
        "session_id": "sess-1",
        "thread_id": "sess-1",
        "decision_id": str(uuid4()),
        "execution_ref": str(uuid4()),
    })
    runner.resume = AsyncMock(return_value={"status": "EXECUTING", "thread_id": "sess-1"})
    runner.get_thread_state = AsyncMock(return_value={
        "thread_id": "sess-1",
        "values": {},
        "next": (),
        "interrupts": [],
    })
    mock_app_state.decision_runner = runner
    return mock_app_state


@pytest.fixture
def mock_app_state():
    state = MagicMock()
    state.decision_runner = None
    return state


@pytest.mark.asyncio
async def test_decision_run_disabled_returns_503():
    from httpx import AsyncClient, ASGITransport

    xnch_app.state = MagicMock()
    xnch_app.state.decision_runner = None

    transport = ASGITransport(app=xnch_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/decision/run",
            json={
                "raw_input": "list services",
                "session_id": str(uuid4()),
                "trace_id": str(uuid4()),
                "actor": {"id": "u1", "role": "OPERATOR", "capability_set": []},
                "system_state_version": "v1",
                "policy_version": "v1",
                "idempotency_key": str(uuid4()),
            },
        )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_decision_run_success(langgraph_app_state):
    from httpx import AsyncClient, ASGITransport

    xnch_app.state = langgraph_app_state

    transport = ASGITransport(app=xnch_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/decision/run",
            json={
                "raw_input": "list services",
                "session_id": "sess-1",
                "trace_id": str(uuid4()),
                "actor": {"id": "u1", "role": "OPERATOR", "capability_set": []},
                "system_state_version": "v1",
                "policy_version": "v1",
                "idempotency_key": str(uuid4()),
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "EXECUTING"
    langgraph_app_state.decision_runner.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_decision_resume(langgraph_app_state):
    from httpx import AsyncClient, ASGITransport

    xnch_app.state = langgraph_app_state

    transport = ASGITransport(app=xnch_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/decision/sess-1/resume",
            json={"payload": True},
        )

    assert response.status_code == 200
    langgraph_app_state.decision_runner.resume.assert_awaited_once_with("sess-1", True)
