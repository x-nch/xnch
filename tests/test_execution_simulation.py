"""xnch simulated execution + goal advancement tests."""
from unittest.mock import AsyncMock, MagicMock, patch

from xnch.routes.execution import (
    ExecutionOutcomeRequest,
    execute_stub,
    execution_outcome,
    simulate_outcome,
)


async def test_simulate_default_is_deterministic():
    assert simulate_outcome("DEPLOY", {"x": 1}) == simulate_outcome("DEPLOY", {"x": 1})


async def test_simulate_default_in_success_failure():
    assert simulate_outcome("DEPLOY", {"x": 1}) in ("SUCCESS", "FAILURE")


def _make_request():
    """Build a FastAPI request mock whose app.state carries the stores."""
    request = MagicMock()
    state = request.app.state
    state.episodic.complete_episode = AsyncMock(return_value="ep-1")
    state.episodic.get_episode_by_decision = AsyncMock(return_value=None)
    state.pg_episodic.complete_decision_episode = AsyncMock(return_value=None)
    state.event_log.emit = MagicMock()
    state.goal_store.complete_step = AsyncMock(return_value={"goal_id": "goal"})
    return request


def _mock_httpx():
    """Patch httpx.AsyncClient with a no-op async context manager client."""
    instance = MagicMock()
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=None)
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    instance.post = AsyncMock(return_value=resp)
    client = MagicMock()
    client.return_value = instance
    return patch("httpx.AsyncClient", client)


async def test_execute_stub_honors_simulation_override():
    """simulation={"outcome": "fail"} must normalize to outcome_status="FAILURE"."""
    body = {
        "execution_ref": "ref-1",
        "decision_id": "dec-1",
        "execution_token": "tok-1",
        "goal_id": "goal-1",
        "action_spec": {"type": "DEPLOY", "params": {"x": 1}},
        "simulation": {"outcome": "fail"},
    }
    request = MagicMock()
    with patch(
        "xnch.routes.execution.execution_outcome",
        new=AsyncMock(return_value={"status": "ok"}),
    ) as mock_outcome:
        await execute_stub(body, request)

    outcome = mock_outcome.call_args.args[0]
    assert outcome.outcome_status == "FAILURE"
    assert outcome.goal_id == "goal-1"


async def test_execute_stub_falls_back_to_hash_when_no_override():
    """Without a simulation override, execute_stub must use simulate_outcome."""
    action_type = "DEPLOY"
    params = {"x": 1}
    body = {
        "execution_ref": "ref-2",
        "decision_id": "dec-2",
        "action_spec": {"type": action_type, "params": params},
    }
    request = MagicMock()
    with patch(
        "xnch.routes.execution.execution_outcome",
        new=AsyncMock(return_value={"status": "ok"}),
    ) as mock_outcome:
        await execute_stub(body, request)

    outcome = mock_outcome.call_args.args[0]
    assert outcome.outcome_status == simulate_outcome(action_type, params)


async def test_execution_outcome_advances_goal_when_goal_id_present():
    """A present goal_id must trigger app.goal_store.complete_step."""
    body = ExecutionOutcomeRequest(
        execution_ref="ref-3",
        decision_id="dec-3",
        outcome_status="FAILURE",
        goal_id="goal-3",
    )
    request = _make_request()

    with _mock_httpx():
        await execution_outcome(body, request)

    request.app.state.goal_store.complete_step.assert_awaited_once_with(
        "goal-3", "FAILURE"
    )


async def test_execution_outcome_skips_goal_when_no_goal_id():
    """An empty goal_id must NOT call complete_step."""
    body = ExecutionOutcomeRequest(
        execution_ref="ref-4",
        decision_id="dec-4",
        outcome_status="SUCCESS",
        goal_id="",
    )
    request = _make_request()

    with _mock_httpx():
        await execution_outcome(body, request)

    request.app.state.goal_store.complete_step.assert_not_awaited()
