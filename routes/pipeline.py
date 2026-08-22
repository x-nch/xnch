"""HITL pipeline API — invoke/resume the LangGraph decision graph."""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from ..agents.pipeline_runtime import PipelineRuntime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/governance/pipeline", tags=["governance"])


class PipelineInvokeRequest(BaseModel):
    session_id: str
    raw_input: str
    thread_id: str | None = None
    trace_id: str | None = None


class PipelineResumeRequest(BaseModel):
    thread_id: str
    decision: Annotated[str | None, Field(pattern="^(approve|reject)$")] = None
    approved: bool | None = None

    @model_validator(mode="after")
    def _require_choice(self) -> "PipelineResumeRequest":
        if self.decision is None and self.approved is None:
            raise ValueError("resume requires decision ('approve'|'reject') or approved bool")
        return self


def _get_runtime(request: Request) -> PipelineRuntime:
    runtime: PipelineRuntime | None = getattr(request.app.state, "pipeline_runtime", None)
    if runtime is None or not runtime.ready:
        raise HTTPException(
            status_code=503,
            detail="LangGraph pipeline disabled (set XNCH_LANGGRAPH_PIPELINE=true)",
        )
    return runtime


@router.post("/invoke")
async def invoke_pipeline(body: PipelineInvokeRequest, request: Request) -> dict[str, Any]:
    """Run a decision through the LangGraph graph; may pause for HITL approval."""
    runtime = _get_runtime(request)
    try:
        return await runtime.invoke(
            raw_input=body.raw_input,
            session_id=body.session_id,
            trace_id=body.trace_id,
            thread_id=body.thread_id,
        )
    except Exception as exc:
        logger.warning("pipeline invoke failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"pipeline invoke failed: {exc}") from exc


@router.post("/resume")
async def resume_pipeline(body: PipelineResumeRequest, request: Request) -> dict[str, Any]:
    """Approve/reject a pending EXECUTION interrupt."""
    runtime = _get_runtime(request)
    try:
        return await runtime.resume(
            thread_id=body.thread_id, decision=body.decision, approved=body.approved
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{thread_id}")
async def get_pending_pipeline(thread_id: str, request: Request) -> dict[str, Any]:
    """Return pending interrupts / graph values for a thread."""
    runtime = _get_runtime(request)
    try:
        return await runtime.get_pending(thread_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"no state for thread {thread_id!r}") from exc
