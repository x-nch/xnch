"""FastAPI router for the beeAI orchestration path.

Mounted only when ``XNCH_BEEAI_ENABLED=true``. Returns 404 otherwise, so the
existing LangGraph / nexi paths are never affected unless the flag is on.

Actor gating mirrors the rest of xnch: ``X-Actor-Role`` / ``X-Trace-Id`` /
``X-Session-Id`` headers build an ``ActorContext`` that the MCP tool registry
already trusts. Mutation approval is operator-gated via ``X-BeeAI-Approval``.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...config import settings
from ...memory.audit_store import emit_event
from xnch_mcp.context import ActorContext

from .backend import StaticChatModel
from .runtime import run_agent, run_swarm

beeai_router = APIRouter(prefix="/beeai", tags=["beeai"])


class BeeaiChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class BeeaiRunResponse(BaseModel):
    engine: str = "beeai"
    text: str
    tool_count: int = 0
    duration_ms: int = 0


def _actor_from_request(request: Request) -> ActorContext:
    role = request.headers.get("X-Actor-Role", "external")
    trace_id = request.headers.get("X-Trace-Id") or str(uuid4())
    session_id = request.headers.get("X-Session-Id")
    return ActorContext(actor_role=role, trace_id=trace_id, session_id=session_id)


def _approval_from_request(request: Request) -> bool:
    return request.headers.get("X-BeeAI-Approval", "").lower() == "allow"


def _ensure_enabled() -> None:
    if not settings.beeai_enabled:
        raise HTTPException(status_code=404, detail="beeai engine disabled")


@beeai_router.get("/health")
async def beeai_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "engine": "beeai",
        "enabled": settings.beeai_enabled,
        "demo_mode": settings.beeai_demo_mode,
        "model": settings.beeai_model,
    }


@beeai_router.post("/chat", response_model=BeeaiRunResponse)
async def beeai_chat(body: BeeaiChatRequest, request: Request) -> BeeaiRunResponse:
    """Run the beeAI orchestrator agent over the in-process MCP tool registry."""
    _ensure_enabled()
    actor = _actor_from_request(request)
    approve = _approval_from_request(request)
    event_log = getattr(request.app.state, "event_log", None)
    llm = StaticChatModel() if settings.beeai_demo_mode else None
    result = await run_agent(
        body.message,
        app_state=request.app.state,
        actor=actor,
        event_log=event_log,
        approve=approve,
        llm=llm,
    )
    emit_event(
        actor.trace_id,
        "xnch.beeai",
        "BEEAI_CHAT",
        {
            "session_id": body.session_id,
            "tool_count": result["tool_count"],
            "duration_ms": result["duration_ms"],
        },
    )
    return BeeaiRunResponse(**result)


@beeai_router.post("/swarm", response_model=BeeaiRunResponse)
async def beeai_swarm(body: BeeaiChatRequest, request: Request) -> BeeaiRunResponse:
    """Run the AgentWorkflow swarm demo (context_bee + planner_bee)."""
    _ensure_enabled()
    actor = _actor_from_request(request)
    approve = _approval_from_request(request)
    event_log = getattr(request.app.state, "event_log", None)
    llm = StaticChatModel() if settings.beeai_demo_mode else None
    result = await run_swarm(
        body.message,
        app_state=request.app.state,
        actor=actor,
        event_log=event_log,
        approve=approve,
        llm=llm,
    )
    emit_event(
        actor.trace_id,
        "xnch.beeai",
        "BEEAI_SWARM",
        {
            "session_id": body.session_id,
            "tool_count": result["tool_count"],
            "duration_ms": result["duration_ms"],
        },
    )
    return BeeaiRunResponse(**result)
