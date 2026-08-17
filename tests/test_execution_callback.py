"""xnch /execution/outcome nexi-callback context enrichment tests."""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from xnch.memory.db import init_db
from xnch.memory.episodic_store import EpisodicStore
from xnch.routes.execution import ExecutionOutcomeRequest


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    return path


async def _make_outcome():
    return ExecutionOutcomeRequest(
        execution_ref=str(uuid4()),
        decision_id="dec-cb-1",
        execution_token_ref="",
        outcome_status="SUCCESS",
        observed_state_delta={},
        side_effects_observed=[],
        duration_ms=50,
        anomalies=[],
    )


async def test_fire_nexi_callback_includes_context_tuple_from_sqlite_store(db_path):
    """Callback payload must carry intent/action/entity/actor from the SQLite episode."""
    await init_db(db_path)
    store = EpisodicStore(db_path)
    await store.create_episode(
        decision_id="dec-cb-1",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="operator",
        context_snapshot={"outcome_score_predicted": 0.8},
    )

    from xnch.routes.execution import _fire_nexi_callback

    app = MagicMock()
    app.episodic = store

    body = await _make_outcome()
    with patch("httpx.AsyncClient") as mock_client:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client.return_value = mock_client_instance

        # Pass a PG episode_id that does NOT exist in SQLite — the callback must
        # still resolve context via decision_id.
        await _fire_nexi_callback(body, "pg-episode-not-in-sqlite", app)

    payload = mock_client_instance.post.call_args.kwargs["json"]
    assert payload["intent_class"] == "EXECUTION"
    assert payload["action_type"] == "DEPLOY"
    assert payload["entity_class"] == "SERVICE"
    assert payload["actor_role"] == "operator"
    assert payload["outcome_score_predicted"] == 0.8


async def test_fire_nexi_callback_defaults_when_no_episode(db_path):
    """Callback should send empty context fields when no episode exists."""
    await init_db(db_path)
    store = EpisodicStore(db_path)

    from xnch.routes.execution import _fire_nexi_callback

    app = MagicMock()
    app.episodic = store

    body = await _make_outcome()

    with patch("httpx.AsyncClient") as mock_client:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client.return_value = mock_client_instance

        await _fire_nexi_callback(body, None, app)

    payload = mock_client_instance.post.call_args.kwargs["json"]
    assert payload["intent_class"] == ""
    assert payload["action_type"] == ""
    assert payload["outcome_score_predicted"] == 0.5
