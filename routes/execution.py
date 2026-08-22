"""Step 11-13: execution dispatch stub and outcome recording."""
import asyncio
import hashlib
import json
import logging
from typing import Any

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/execution", tags=["execution"])


class ExecutionOutcomeRequest(BaseModel):
    execution_ref: str
    decision_id: str
    execution_token_ref: str = ""
    outcome_status: str
    observed_state_delta: dict[str, Any] = {}
    side_effects_observed: list[str] = []
    duration_ms: int = 0
    anomalies: list[str] = []
    goal_id: str = ""


def simulate_outcome(action_type: str, params: dict[str, Any]) -> str:
    """Deterministically simulate an execution outcome from (action_type, params)."""
    digest = hashlib.sha256(
        json.dumps({"action_type": action_type, "params": params}, sort_keys=True).encode()
    ).hexdigest()
    return "SUCCESS" if int(digest[:2], 16) % 2 == 0 else "FAILURE"


def _normalize_outcome(value: str) -> str:
    """Normalize a simulation override to the UPPERCASE complete_step form.

    Maps ``success→SUCCESS``, ``fail→FAILURE``, ``partial→PARTIAL``
    (case-insensitive); ``FAIL`` is the only alias that does not map 1:1.
    """
    upper = value.strip().upper()
    return "FAILURE" if upper == "FAIL" else upper


@router.post("/execute")
async def execute_stub(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Stub execution runner — resolves a simulation override or a deterministic hash."""
    action_spec = body.get("action_spec") or {}
    sim = body.get("simulation") or {}
    outcome_override = sim.get("outcome")
    if outcome_override is not None:
        status = _normalize_outcome(str(outcome_override))
    else:
        status = simulate_outcome(
            action_spec.get("type", ""), action_spec.get("params", {}) or {}
        )
    outcome = ExecutionOutcomeRequest(
        execution_ref=str(body.get("execution_ref", "")),
        decision_id=str(body.get("decision_id", "")),
        execution_token_ref=str(body.get("execution_token", "")),
        outcome_status=status,
        goal_id=str(body.get("goal_id") or ""),
        duration_ms=50,
    )
    return await execution_outcome(outcome, request)


@router.post("/outcome")
async def execution_outcome(body: ExecutionOutcomeRequest, request: Request) -> dict[str, Any]:
    """Step 13: write episode outcome then fire async Nexi callback."""
    app = request.app.state

    episode_id = await app.episodic.complete_episode(
        decision_id=body.decision_id,
        outcome=body.outcome_status,
        observed_state_delta=body.observed_state_delta,
        side_effects=body.side_effects_observed,
        duration_ms=body.duration_ms,
        anomalies=body.anomalies,
    )

    pg_episode_id = await app.pg_episodic.complete_decision_episode(
        decision_id=body.decision_id,
        outcome=body.outcome_status,
    )
    if pg_episode_id:
        episode_id = str(pg_episode_id)

    app.event_log.emit(
        body.decision_id, "xnch.execution", "OUTCOME_RECORDED",
        data={"episode_id": str(episode_id), "outcome": body.outcome_status},
    )

    asyncio.create_task(
        _fire_nexi_callback(body, episode_id, app)
    )

    if body.goal_id:
        try:
            await app.goal_store.complete_step(body.goal_id, body.outcome_status)
        except Exception as exc:
            logger.error("goal advance failed (goal_id=%s): %s", body.goal_id, exc)

    return {"status": "ok", "episode_id": episode_id}


async def _fire_nexi_callback(
    body: ExecutionOutcomeRequest,
    episode_id: str | None,
    app,
) -> None:
    outcome_score_predicted = 0.5
    intent_class = getattr(body, "intent_class", "")
    action_type = getattr(body, "action_type", "")
    entity_class = getattr(body, "entity_class", "")
    actor_role = getattr(body, "actor_role", "")
    # Context tuple lives in the SQLite episodic store keyed by decision_id
    # (written at verdict time). Look it up by decision_id, not by episode_id,
    # because episode_id may be the PG id which does not exist in SQLite.
    ep = await app.episodic.get_episode_by_decision(body.decision_id)
    if ep and ep.get("context_snapshot"):
        snap = json.loads(ep["context_snapshot"])
        outcome_score_predicted = snap.get("outcome_score_predicted", 0.5)
    if ep:
        intent_class = ep.get("intent_class", intent_class)
        action_type = ep.get("action_type", action_type)
        entity_class = ep.get("entity_class", entity_class)
        actor_role = ep.get("actor_role", actor_role)

    payload = {
        "execution_ref": body.execution_ref,
        "decision_id": body.decision_id,
        "episode_id": str(episode_id) if episode_id else "",
        "outcome_status": body.outcome_status,
        "outcome_score_predicted": outcome_score_predicted,
        "intent_class": intent_class,
        "action_type": action_type,
        "entity_class": entity_class,
        "actor_role": actor_role,
        "trace_id": body.decision_id,
    }
    try:
        async with httpx.AsyncClient(base_url=settings.nexi_base_url, timeout=10.0) as client:
            resp = await client.post("/callback/outcome", json=payload)
            resp.raise_for_status()
    except Exception as exc:
        logger.error("Nexi callback failed (decision_id=%s): %s", body.decision_id, exc)
