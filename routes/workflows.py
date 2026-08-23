"""Workflow CRUD + run + unified approval queue endpoints.

Backed by app.state.workflow_store. Write endpoints are gated by the
Hybrid-B gateway token (see security/gateway_token.py): when
``app.state.gateway_secret`` is set, requests must present a valid
``X-Gateway-Token`` or ``X-Service-Key``; when unset (dev/test) the gate is open.

Imports stay light (fastapi/pydantic/models/gateway_token only) so this module
is loadable without the full xnch dependency tree.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Literal

from xnch.models.workflow import (
    ApprovalDecideRequest,
    WorkflowCreateRequest,
    WorkflowUpdateRequest,
)
from xnch.security.gateway_token import verify_gateway_token, verify_service_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["workflows"])
approvals_router = APIRouter(prefix="/approvals", tags=["approvals"])


async def require_gateway_access(
    request: Request,
    x_gateway_token: str | None = Header(default=None),
    x_service_key: str | None = Header(default=None),
) -> None:
    secret = getattr(request.app.state, "gateway_secret", "")
    if not secret:
        return
    if verify_gateway_token(secret, x_gateway_token or ""):
        return
    if verify_service_key(secret, x_service_key):
        return
    raise HTTPException(status_code=401, detail="gateway token required")


def _serialize_workflow(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["id"] = out.pop("id", None)
    for col in ("trigger_json", "steps_json"):
        raw = out.pop(col, None)
        if isinstance(raw, str):
            try:
                out[col.removesuffix("_json")] = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                out[col.removesuffix("_json")] = [] if col == "steps_json" else {}
    return out


def _serialize_run(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for col in ("trigger_json", "steps_json"):
        raw = out.pop(col, None)
        if isinstance(raw, str):
            try:
                out[col.removesuffix("_json")] = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                out[col.removesuffix("_json")] = []
    return out


def _serialize_approval(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    raw = out.pop("payload_json", None)
    if isinstance(raw, str):
        try:
            out["payload"] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            out["payload"] = {}
    return out


def _sync_schedule_job(request: Request, row: dict[str, Any] | None) -> None:
    """Keep the APScheduler job in sync after workflow CRUD (P3)."""
    if row is None:
        return
    scheduler = getattr(request.app.state, "scheduler", None)
    store = request.app.state.workflow_store
    if scheduler is None:
        return
    from ..jobs.workflow_schedule import sync_workflow_job

    sync_workflow_job(scheduler, row, store=store)


def _get_store(request: Request):
    store = getattr(request.app.state, "workflow_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="workflow_store unavailable")
    return store


# ----------------------------------------------------------------------
# Workflows
# ----------------------------------------------------------------------


@router.post("/workflows", status_code=201, dependencies=[Depends(require_gateway_access)])
async def create_workflow(
    body: WorkflowCreateRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    store = _get_store(request)
    wf_id = await store.create_workflow(
        owner_actor_id=body.owner_actor_id,
        name=body.name,
        description=body.description,
        trigger=body.trigger.model_dump(),
        steps=[s.model_dump() for s in body.steps],
    )
    row = await store.get_workflow(wf_id)
    assert row is not None
    _sync_schedule_job(request, row)
    return _serialize_workflow(row)


@router.get("/workflows")
async def list_workflows(request: Request) -> list[dict[str, Any]]:
    return [_serialize_workflow(r) for r in await _get_store(request).list_workflows()]


@router.patch("/workflows/{workflow_id}", dependencies=[Depends(require_gateway_access)])
async def update_workflow(
    workflow_id: str,
    body: WorkflowUpdateRequest,
    request: Request,
) -> dict[str, Any]:
    patch = body.model_dump(exclude_none=True)
    row = await _get_store(request).update_workflow(workflow_id, patch)
    if not row:
        raise HTTPException(status_code=404, detail="workflow not found")
    _sync_schedule_job(request, row)
    return _serialize_workflow(row)


@router.delete("/workflows/{workflow_id}", status_code=204, dependencies=[Depends(require_gateway_access)])
async def delete_workflow(workflow_id: str, request: Request) -> None:
    deleted = await _get_store(request).delete_workflow(workflow_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="workflow not found")
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None:
        from ..jobs.workflow_schedule import remove_workflow_job

        remove_workflow_job(scheduler, workflow_id)


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------


@router.post("/workflows/{workflow_id}/run", status_code=201, dependencies=[Depends(require_gateway_access)])
async def run_workflow(
    workflow_id: str,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    store = _get_store(request)
    try:
        run, created = await store.create_run(
            workflow_id=workflow_id,
            actor=request.headers.get("X-Actor-Id", "operator"),
            idempotency_key=idempotency_key,
        )
    except Exception as exc:
        if type(exc).__name__ == "WorkflowNotFound":
            raise HTTPException(status_code=404, detail="workflow not found") from exc
        raise
    out = _serialize_run(run)
    out["created"] = created
    if not created:
        out["detail"] = "idempotent replay — existing run returned"
    return out


@router.get("/workflows/runs")
async def list_runs(
    request: Request,
    status: str | None = None,
    workflow_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return [
        _serialize_run(r)
        for r in await _get_store(request).list_runs(
            status=status, workflow_id=workflow_id, limit=limit
        )
    ]


@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str, request: Request) -> dict[str, Any]:
    row = await _get_store(request).get_workflow(workflow_id)
    if not row:
        raise HTTPException(status_code=404, detail="workflow not found")
    return _serialize_workflow(row)


class StepClaimRequest(BaseModel):
    lease_owner: str
    ttl_s: int = 120


class StepOutcomeRequest(BaseModel):
    outcome_status: Literal["SUCCESS", "PARTIAL", "FAILURE"]


@router.post(
    "/workflows/steps/claim",
    dependencies=[Depends(require_gateway_access)],
    response_model=None,
)
async def claim_step(body: StepClaimRequest, request: Request):
    store = _get_store(request)
    step = await store.claim_next_approved_step(
        lease_owner=body.lease_owner, ttl_s=body.ttl_s
    )
    if step is None:
        return Response(status_code=204)
    payload = json.loads(step.get("payload_json") or "{}")
    return {**step, "payload": payload}


@router.post(
    "/workflows/steps/{step_uuid}/outcome",
    dependencies=[Depends(require_gateway_access)],
)
async def step_outcome(
    step_uuid: str,
    body: StepOutcomeRequest,
    request: Request,
) -> dict[str, Any]:
    store = _get_store(request)
    try:
        row = await store.complete_step(
            step_uuid,
            outcome_status=body.outcome_status,
            actor=request.headers.get("X-Actor-Id", "nexi"),
        )
    except Exception as exc:
        if type(exc).__name__ == "WorkflowNotFound":
            raise HTTPException(status_code=404, detail="step not found") from exc
        raise
    payload = json.loads(row.get("payload_json") or "{}")
    return {**row, "payload": payload}


# ----------------------------------------------------------------------
# Approvals — unified queue
# ----------------------------------------------------------------------


@approvals_router.get("")
async def list_approvals(
    request: Request,
    status: str | None = "pending",
    producer_type: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    rows = await _get_store(request).list_approvals(
        status=status, producer_type=producer_type, limit=limit
    )
    return [_serialize_approval(r) for r in rows]


@approvals_router.get("/{approval_id}")
async def get_approval(approval_id: str, request: Request) -> dict[str, Any]:
    row = await _get_store(request).get_approval(approval_id)
    if not row:
        raise HTTPException(status_code=404, detail="approval not found")
    return _serialize_approval(row)


@approvals_router.post("/{approval_id}/decide", dependencies=[Depends(require_gateway_access)])
async def decide_approval(
    approval_id: str,
    body: ApprovalDecideRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    store = _get_store(request)
    actor = request.headers.get("X-Actor-Id", "operator")
    try:
        row = await store.decide_approval(
            approval_id,
            decision=body.decision,
            actor=actor,
            note=body.note,
            idempotency_key=idempotency_key,
        )
    except Exception as exc:
        if type(exc).__name__ == "WorkflowNotFound":
            raise HTTPException(status_code=404, detail="approval not found") from exc
        if type(exc).__name__ == "ApprovalConflict":
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise
    if row.get("producer_type") == "goal_step":
        from xnch.jobs.goal_dispatch import spawn_agent_run_for_approval

        runner_store = getattr(request.app.state, "agent_run_store", None)
        goal_store = getattr(request.app.state, "goal_store", None)
        try:
            if body.decision == "approve" and runner_store is not None:
                await spawn_agent_run_for_approval(
                    agent_run_store=runner_store, approval=row,
                    workflow_store=getattr(request.app.state, "workflow_store", None),
                )
            elif body.decision == "reject" and goal_store is not None:
                await goal_store.complete_step(row["producer_id"], "FAILURE")
        except Exception:  # noqa: BLE001 — decision already recorded; side-effect failures log only
            import logging

            logging.getLogger(__name__).exception(
                "goal_step post-decide hook failed for %s", row["id"]
            )
    return _serialize_approval(row)


@approvals_router.get("/{approval_id}/events")
async def approval_events(approval_id: str, request: Request) -> list[dict[str, Any]]:
    store = _get_store(request)
    row = await store.get_approval(approval_id)
    if not row:
        raise HTTPException(status_code=404, detail="approval not found")
    payload = json.loads(row.get("payload_json") or "{}")
    step_uuid = payload.get("run_id") and row.get("producer_id")
    if not step_uuid:
        return []
    return await store.step_events(str(step_uuid))
