"""LangGraph decision pipeline endpoints."""
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/decision", tags=["decision"])


class DecisionRunRequest(BaseModel):
    raw_input: str
    session_id: str
    trace_id: str
    actor: dict[str, Any]
    system_state_version: str
    policy_version: str
    idempotency_key: str
    priority: str = "NORMAL"


class DecisionResumeRequest(BaseModel):
    payload: Any = Field(default=True)


def _require_runner(request: Request):
    runner = request.app.state.decision_runner
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="LangGraph pipeline disabled. Set XNCH_USE_LANGGRAPH=true.",
        )
    return runner


@router.post("/run")
async def run_decision(body: DecisionRunRequest, request: Request) -> dict[str, Any]:
    """Run the LangGraph decision pipeline for a session."""
    runner = _require_runner(request)
    return await runner.run(
        raw_input=body.raw_input,
        session_id=body.session_id,
        trace_id=body.trace_id,
        actor=body.actor,
        system_state_version=body.system_state_version,
        policy_version=body.policy_version,
        idempotency_key=body.idempotency_key,
        priority=body.priority,
    )


@router.post("/{thread_id}/resume")
async def resume_decision(
    thread_id: str,
    body: DecisionResumeRequest,
    request: Request,
) -> dict[str, Any]:
    """Resume a paused graph after human-in-the-loop interrupt."""
    runner = _require_runner(request)
    return await runner.resume(thread_id, body.payload)


@router.get("/{thread_id}")
async def get_decision_state(thread_id: str, request: Request) -> dict[str, Any]:
    """Return checkpointed graph state for a decision thread."""
    runner = _require_runner(request)
    return await runner.get_thread_state(thread_id)
