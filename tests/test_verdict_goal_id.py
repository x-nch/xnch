"""xnch /verdict context_snapshot goal_id threading test."""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from xnch.routes.verdict import VerdictRequest, verdict


def _make_app() -> MagicMock:
    """Build a MagicMock app exposing every attribute the verdict route touches."""
    app = MagicMock()
    app.get_state_version = AsyncMock(return_value="v1.0.0")
    app.get_policy_version = AsyncMock(return_value="p1.0.0")

    resolved = MagicMock()
    resolved.id = "act-1"
    resolved.role = "AGENT"
    resolved.capability_set = []
    app.governance.resolve_actor = AsyncMock(return_value=resolved)

    result = MagicMock()
    result.verdict = "ALLOW"
    result.policy_refs = []
    result.modified_action_spec = None
    app.policy_engine.evaluate = MagicMock(return_value=result)

    app.ledger.write = MagicMock()
    app.event_log.emit = MagicMock()
    app.token_signer.issue = MagicMock(return_value=("token-abc", 3600))

    app.episodic.create_episode = AsyncMock()
    app.pg_episodic.store_decision_episode = AsyncMock()
    return app


def _make_body(goal_id: str | None) -> VerdictRequest:
    ctx: dict[str, object] = {
        "session_id": "sess-1",
        "system_state_version": "v1.0.0",
        "outcome_score_predicted": 0.8,
    }
    if goal_id is not None:
        ctx["goal_id"] = goal_id
    return VerdictRequest(
        request_id="req-1",
        actor={"id": "act-1", "claimed_role": "AGENT"},
        action={
            "type": "DEPLOY",
            "target": "svc",
            "payload_hash": "h",
            "payload": {},
            "intent_class": "EXECUTION",
            "entity_class": "SERVICE",
        },
        context=ctx,
    )


async def test_verdict_context_snapshot_carries_goal_id():
    """The episode context_snapshot must persist goal_id from the request context."""
    app = _make_app()
    request = MagicMock()
    request.app.state = app

    goal_id = str(uuid4())
    body = _make_body(goal_id)

    with patch("xnch.routes.verdict.trace_llm_call", new=AsyncMock()):
        await verdict(body, request)

    snapshot = app.episodic.create_episode.call_args.kwargs["context_snapshot"]
    assert snapshot["goal_id"] == goal_id
    assert snapshot["session_id"] == "sess-1"
    assert snapshot["actor_id"] == "act-1"
    assert snapshot["outcome_score_predicted"] == 0.8


async def test_verdict_context_snapshot_goal_id_defaults_to_empty():
    """Without a goal_id in context, the snapshot must carry an empty string."""
    app = _make_app()
    request = MagicMock()
    request.app.state = app

    body = _make_body(None)

    with patch("xnch.routes.verdict.trace_llm_call", new=AsyncMock()):
        await verdict(body, request)

    snapshot = app.episodic.create_episode.call_args.kwargs["context_snapshot"]
    assert snapshot["goal_id"] == ""
