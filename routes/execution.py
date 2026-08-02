"""Step 11-13: execution dispatch stub and outcome recording."""
import asyncio
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


@router.post("/execute")
async def execute_stub(body: dict[str, Any], request: Request) -> dict[str, Any]:
    """Stub execution runner — records SUCCESS and completes the decision episode."""
    outcome = ExecutionOutcomeRequest(
        execution_ref=str(body.get("execution_ref", "")),
        decision_id=str(body.get("decision_id", "")),
        execution_token_ref=str(body.get("execution_token", "")),
        outcome_status="SUCCESS",
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
        episode_id = pg_episode_id

    app.event_log.emit(
        body.decision_id, "xnch.execution", "OUTCOME_RECORDED",
        data={"episode_id": episode_id, "outcome": body.outcome_status},
    )

    asyncio.create_task(
        _fire_nexi_callback(body, episode_id, app)
    )

    return {"status": "ok", "episode_id": episode_id}


async def _fire_nexi_callback(
    body: ExecutionOutcomeRequest,
    episode_id: str | None,
    app,
) -> None:
    outcome_score_predicted = 0.5
    if episode_id:
        ep = await app.episodic.get_episode(episode_id)
        if ep and ep.get("context_snapshot"):
            snap = json.loads(ep["context_snapshot"])
            outcome_score_predicted = snap.get("outcome_score_predicted", 0.5)

    payload = {
        "execution_ref": body.execution_ref,
        "decision_id": body.decision_id,
        "episode_id": episode_id,
        "outcome_status": body.outcome_status,
        "outcome_score_predicted": outcome_score_predicted,
        "trace_id": body.decision_id,
    }
    try:
        async with httpx.AsyncClient(base_url=settings.nexi_base_url, timeout=10.0) as client:
            resp = await client.post("/callback/outcome", json=payload)
            resp.raise_for_status()
    except Exception as exc:
        logger.error("Nexi callback failed (decision_id=%s): %s", body.decision_id, exc)
