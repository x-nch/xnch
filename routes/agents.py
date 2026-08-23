"""Agent dispatch routes — queue tasks for external coding-agent runners.

Pull model: a Mac-side runner claims QUEUED runs via /agents/dispatch/next
(lease-based, same protocol family as the workflow executor) and pushes an
outcome back. Writes are Hybrid-B gateway-token gated; reads are open.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from xnch.models.agent import (
    AgentClaimRequest,
    AgentDispatchRequest,
    AgentOutcomeRequest,
)
from xnch.routes.workflows import require_gateway_access

router = APIRouter(prefix="/agents", tags=["agents"])


def _get_store(request: Request):
    store = getattr(request.app.state, "agent_run_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="agent_run_store unavailable")
    return store


@router.post("/dispatch", status_code=201, dependencies=[Depends(require_gateway_access)])
async def dispatch_agent_task(
    body: AgentDispatchRequest,
    request: Request,
) -> dict[str, Any]:
    return await _get_store(request).create_run(prompt=body.prompt, workspace=body.workspace)


@router.post("/dispatch/next", dependencies=[Depends(require_gateway_access)])
async def claim_next_agent_run(body: AgentClaimRequest, request: Request):
    store = _get_store(request)
    row = await store.claim_next(body.runner_id, ttl_s=body.ttl_s)
    if row is None:
        return Response(status_code=204)
    return row


@router.post("/runs/{run_id}/outcome", dependencies=[Depends(require_gateway_access)])
async def agent_run_outcome(
    run_id: str,
    body: AgentOutcomeRequest,
    request: Request,
) -> dict[str, Any]:
    store = _get_store(request)
    row = await store.complete_run(
        run_id,
        outcome_status=body.outcome_status,
        exit_code=body.exit_code,
        output_path=body.output_path,
        error=body.error,
    )
    if row is None:
        existing = await store.get_run(run_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="agent run not found")
        raise HTTPException(
            status_code=409,
            detail=f"agent run is {existing['status']}, not RUNNING",
        )
    from xnch.jobs.goal_dispatch import apply_outcome_backpressure

    try:
        await apply_outcome_backpressure(
            agent_run_store=store,
            workflow_store=getattr(request.app.state, "workflow_store", None),
            goal_store=getattr(request.app.state, "goal_store", None),
            run_row=row,
        )
    except Exception:  # noqa: BLE001 — outcome recorded; back-pressure logs only
        import logging

        logging.getLogger(__name__).exception(
            "goal step-outcome back-pressure failed for run %s", run_id
        )
    return row


@router.get("/runs")
async def list_agent_runs(
    request: Request,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return await _get_store(request).list_runs(status=status, limit=limit)
