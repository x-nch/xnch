"""Goal CRUD + claim endpoints backed by app.state.goal_store.

Every endpoint that returns a goal serializes the raw DB row first:
``simulation_plan`` is stored as a JSON string, but the wire contract (the
``Goal`` Pydantic model) expects ``list[dict]``. ``_serialize_goal`` performs
that conversion and drops the storage-only ``schema_version`` column.
"""
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/goals", tags=["goals"])


class GoalCreateRequest(BaseModel):
    owner_actor_id: str
    objective: str
    max_steps: int = 10
    failure_threshold: int = 3
    simulation_plan: list[dict[str, Any]] | None = None


class GoalClaimRequest(BaseModel):
    lease_owner: str


class GoalUpdateRequest(BaseModel):
    status: str | None = None
    progress: str | None = None


class GoalStepOutcomeRequest(BaseModel):
    outcome_status: str


def _serialize_goal(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw DB row into the wire shape for the ``Goal`` model."""
    out = dict(row)
    sp = out.get("simulation_plan")
    if isinstance(sp, str):
        try:
            out["simulation_plan"] = json.loads(sp) if sp else []
        except json.JSONDecodeError:
            out["simulation_plan"] = []
    out.pop("schema_version", None)
    return out


@router.post("")
async def create_goal(body: GoalCreateRequest, request: Request) -> dict[str, Any]:
    app = request.app.state
    goal_id = await app.goal_store.create_goal(
        owner_actor_id=body.owner_actor_id,
        objective=body.objective,
        max_steps=body.max_steps,
        failure_threshold=body.failure_threshold,
        simulation_plan=body.simulation_plan,
    )
    row = await app.goal_store.get_goal(goal_id)
    return _serialize_goal(row)


@router.get("")
async def list_goals(request: Request, status: str | None = None) -> list[dict[str, Any]]:
    app = request.app.state
    rows = await app.goal_store.list_goals(status)
    return [_serialize_goal(r) for r in rows]


@router.get("/{goal_id}")
async def get_goal(goal_id: str, request: Request) -> dict[str, Any]:
    app = request.app.state
    row = await app.goal_store.get_goal(goal_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Goal not found: {goal_id}")
    return _serialize_goal(row)


@router.post("/claim")
async def claim_goal(body: GoalClaimRequest, request: Request) -> dict[str, Any] | None:
    app = request.app.state
    row = await app.goal_store.claim_next_goal(body.lease_owner)
    return _serialize_goal(row) if row else None


@router.post("/{goal_id}/update")
async def update_goal(goal_id: str, body: GoalUpdateRequest, request: Request) -> dict[str, Any]:
    app = request.app.state
    row = await app.goal_store.update_goal(
        goal_id, status=body.status, progress=body.progress
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Goal not found: {goal_id}")
    return _serialize_goal(row)


@router.post("/{goal_id}/step-outcome")
async def step_outcome(goal_id: str, body: GoalStepOutcomeRequest, request: Request) -> dict[str, Any]:
    app = request.app.state
    row = await app.goal_store.complete_step(goal_id, body.outcome_status)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Goal not found: {goal_id}")
    return _serialize_goal(row)


@router.post("/{goal_id}/cancel")
async def cancel_goal(goal_id: str, request: Request) -> dict[str, Any]:
    app = request.app.state
    row = await app.goal_store.update_goal(goal_id, status="CANCELLED")
    if row is None:
        raise HTTPException(status_code=404, detail=f"Goal not found: {goal_id}")
    return _serialize_goal(row)
