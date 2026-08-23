"""Phase B alerting instrumentation: HITL gate-bypass counter + Alertmanager webhook receiver."""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY


def _sample_value(name: str, labels: dict[str, str] | None = None) -> float | None:
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name != name:
                continue
            if labels is None or all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return None


# ---------------------------------------------------------------------------
# Goal-loop HITL bypass signal on the verdict path
# ---------------------------------------------------------------------------


def _make_app() -> MagicMock:
    """MagicMock app exposing every attribute the verdict route touches."""
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


def _verdict_body(goal_id: str | None, intent_class: str = "EXECUTION") -> Any:
    from xnch.routes.verdict import VerdictRequest

    ctx: dict[str, object] = {
        "session_id": "sess-1",
        "system_state_version": "v1.0.0",
        "outcome_score_predicted": 0.8,
    }
    if goal_id is not None:
        ctx["goal_id"] = goal_id
    return VerdictRequest(
        request_id=f"req-{uuid4()}",
        actor={"id": "act-1", "claimed_role": "AGENT"},
        action={
            "type": "DEPLOY",
            "target": "svc",
            "payload_hash": "h",
            "payload": {},
            "intent_class": intent_class,
            "entity_class": "SERVICE",
        },
        context=ctx,
    )


async def test_goal_loop_execution_allow_counts_as_bypass() -> None:
    from xnch.routes.verdict import verdict

    before = _sample_value("xnch_hitl_gate_bypass_total", {"origin": "goal_loop"}) or 0.0
    app = _make_app()
    request = MagicMock()
    request.app.state = app

    with pytest.MonkeyPatch.context() as mp:
        import xnch.routes.verdict as vmod

        mp.setattr(vmod, "trace_llm_call", AsyncMock())
        await verdict(_verdict_body(str(uuid4())), request)

    assert (_sample_value("xnch_hitl_gate_bypass_total", {"origin": "goal_loop"}) or 0.0) == before + 1.0
    emitted = [c.args for c in app.event_log.emit.call_args_list]
    assert any(e[2] == "HITL_GATE_BYPASS" for e in emitted)


async def test_chat_origin_execution_does_not_count_as_bypass() -> None:
    from xnch.routes.verdict import verdict

    before = _sample_value("xnch_hitl_gate_bypass_total", {"origin": "goal_loop"}) or 0.0
    app = _make_app()
    request = MagicMock()
    request.app.state = app

    with pytest.MonkeyPatch.context() as mp:
        import xnch.routes.verdict as vmod

        mp.setattr(vmod, "trace_llm_call", AsyncMock())
        await verdict(_verdict_body(None), request)

    assert (_sample_value("xnch_hitl_gate_bypass_total", {"origin": "goal_loop"}) or 0.0) == before


async def test_non_execution_intent_never_counts_as_bypass() -> None:
    from xnch.routes.verdict import verdict

    before = _sample_value("xnch_hitl_gate_bypass_total", {"origin": "goal_loop"}) or 0.0
    app = _make_app()
    request = MagicMock()
    request.app.state = app

    with pytest.MonkeyPatch.context() as mp:
        import xnch.routes.verdict as vmod

        mp.setattr(vmod, "trace_llm_call", AsyncMock())
        await verdict(_verdict_body(str(uuid4()), intent_class="QUERY"), request)

    assert (_sample_value("xnch_hitl_gate_bypass_total", {"origin": "goal_loop"}) or 0.0) == before


async def test_blocked_verdict_does_not_count_as_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    import xnch.routes.verdict as vmod
    from xnch.routes.verdict import verdict

    before = _sample_value("xnch_hitl_gate_bypass_total", {"origin": "goal_loop"}) or 0.0
    app = _make_app()
    blocked = MagicMock()
    blocked.verdict = "BLOCK"
    blocked.policy_refs = ["rule-x"]
    blocked.modified_action_spec = None
    app.policy_engine.evaluate = MagicMock(return_value=blocked)
    request = MagicMock()
    request.app.state = app

    monkeypatch.setattr(vmod, "trace_llm_call", AsyncMock())
    resp = await verdict(_verdict_body(str(uuid4())), request)

    assert resp["verdict"] == "BLOCK"
    assert (_sample_value("xnch_hitl_gate_bypass_total", {"origin": "goal_loop"}) or 0.0) == before


# ---------------------------------------------------------------------------
# Alertmanager webhook receiver (/admin/alerts) + recent-alerts surface
# ---------------------------------------------------------------------------


def _am_payload(status: str = "firing", alertname: str = "HitlGateBypassFiring") -> dict[str, Any]:
    return {
        "version": "4",
        "groupKey": "g1",
        "status": status,
        "receiver": "xnch-webhook",
        "alerts": [
            {
                "status": status,
                "labels": {
                    "alertname": alertname,
                    "severity": "security",
                    "area": "hitl",
                },
                "annotations": {"summary": "EXECUTION decision dispatched outside the gate"},
                "startsAt": "2026-08-23T00:00:00Z",
                "endsAt": "0001-01-01T00:00:00Z" if status == "firing" else "2026-08-23T01:00:00Z",
                "fingerprint": f"fp-{uuid4().hex[:8]}",
            }
        ],
    }


@pytest.fixture
def alerts_app():
    from collections import deque

    from xnch.config import settings

    settings_snapshot = dict(settings.__dict__)
    from xnch.routes.admin import router as admin_router

    app = FastAPI()
    app.include_router(admin_router)
    app.state.event_log = MagicMock()
    app.state.recent_alerts = deque(maxlen=200)
    yield app
    settings.__dict__.clear()
    settings.__dict__.update(settings_snapshot)


@pytest.mark.asyncio
async def test_webhook_accepts_alert_payload_and_records(alerts_app: FastAPI) -> None:
    transport = ASGITransport(app=alerts_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/admin/alerts", json=_am_payload())

    assert resp.status_code == 200
    assert resp.json()["received"] == 1
    assert alerts_app.state.recent_alerts[0]["labels"]["alertname"] == "HitlGateBypassFiring"
    emitted = alerts_app.state.event_log.emit.call_args
    assert emitted.args[2] == "ALERT_FIRING"


@pytest.mark.asyncio
async def test_webhook_resolves_alerts_and_recent_surface(alerts_app: FastAPI) -> None:
    transport = ASGITransport(app=alerts_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        await client.post("/admin/alerts", json=_am_payload(status="firing"))
        await client.post("/admin/alerts", json=_am_payload(status="resolved"))

        recent = await client.get("/admin/alerts/recent")

    body = recent.json()
    assert len(body["alerts"]) == 2
    assert body["alerts"][0]["status"] == "resolved"
    statuses = [a["labels"]["alertname"] for a in body["alerts"]]
    assert statuses.count("HitlGateBypassFiring") == 2


@pytest.mark.asyncio
async def test_webhook_denied_for_non_internal_client(
    alerts_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xnch.config import settings

    monkeypatch.setattr(settings, "alert_webhook_allow_cidrs", ["192.168.50.0/24"])
    transport = ASGITransport(app=alerts_app, client=("203.0.113.9", 5555))
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/admin/alerts", json=_am_payload())

    assert resp.status_code == 403
    assert len(alerts_app.state.recent_alerts) == 0


@pytest.mark.asyncio
async def test_webhook_rejects_malformed_payload(alerts_app: FastAPI) -> None:
    transport = ASGITransport(app=alerts_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/admin/alerts", json={"alerts": "not-a-list"})
        assert resp.status_code == 422

        resp2 = await client.post(
            "/admin/alerts", content=b"{not-json", headers={"content-type": "application/json"}
        )
        assert resp2.status_code == 422
