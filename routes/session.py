"""Step 1-2: Input Layer → xnch → Nexi session initialization."""
import json
import logging
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/session", tags=["session"])


class SessionInitRequest(BaseModel):
    auth_token: str
    raw_input: str
    input_type: str = "TEXT"
    priority: str = "NORMAL"
    source_system: str = ""
    trace_id: str | None = None
    idempotency_key: str | None = None


class ClarifyRequest(BaseModel):
    amended_input: str


async def _run_langgraph_session(
    app,
    session_context: dict[str, Any],
    trace_id: str,
) -> dict[str, Any]:
    runner = app.decision_runner
    if runner is None:
        raise HTTPException(status_code=503, detail="LangGraph pipeline unavailable")

    return await runner.run(
        raw_input=session_context["raw_input"],
        session_id=session_context["session_id"],
        trace_id=trace_id,
        actor=session_context["actor"],
        system_state_version=session_context["system_state_version"],
        policy_version=session_context["policy_version"],
        idempotency_key=session_context["idempotency_key"],
        priority=session_context["priority"],
    )


async def _forward_to_nexi(session_context: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=settings.nexi_base_url, timeout=120.0) as client:
        resp = await client.post("/session/start", json=session_context)
    return resp.json()


@router.post("/init")
async def session_init(body: SessionInitRequest, request: Request) -> dict[str, Any]:
    """Step 1-2: Transport validation, dedup check, actor resolution, then forward to Nexi."""
    app = request.app.state

    trace_id = body.trace_id or str(uuid4())
    idempotency_key = body.idempotency_key or str(uuid4())

    app.event_log.emit(trace_id, "xnch.session", "SESSION_INIT_RECEIVED",
                       data={"source": body.source_system})

    # Step 2a: KV cache dedup check
    cached = await app.kv_cache.get_session(idempotency_key)
    if cached:
        app.event_log.emit(trace_id, "xnch.session", "SESSION_DEDUP_HIT")
        return cached

    # Step 2a: Rate limit (actor_id from unverified token claim — first-line defence)
    unverified_actor = app.token_verifier.verify_bearer(body.auth_token) or "anonymous"
    within_limit = await app.kv_cache.check_rate_limit(unverified_actor)
    if not within_limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Step 2: Auth + actor resolution
    actor_id = app.token_verifier.verify_bearer(body.auth_token)
    if not actor_id:
        raise HTTPException(status_code=401, detail="Invalid auth token")

    actor = await app.governance.resolve_actor(actor_id)
    if not actor:
        raise HTTPException(status_code=401, detail=f"Unknown actor: {actor_id}")

    state_version = await app.get_state_version()
    policy_version = await app.get_policy_version()

    session_context = {
        "session_id": str(uuid4()),
        "trace_id": trace_id,
        "actor": actor.to_dict(),
        "system_state_version": state_version,
        "policy_version": policy_version,
        "idempotency_key": idempotency_key,
        "raw_input": body.raw_input,
        "priority": body.priority,
    }

    await app.kv_cache.set_session(idempotency_key, session_context)
    app.event_log.emit(trace_id, "xnch.session", "SESSION_CREATED",
                       data={"session_id": session_context["session_id"], "actor": actor_id})

    # Forward to Nexi (Step 2 → Step 3) or run LangGraph pipeline
    try:
        if settings.use_langgraph:
            nexi_response = await _run_langgraph_session(app, session_context, trace_id)
        else:
            nexi_response = await _forward_to_nexi(session_context)
    except HTTPException:
        raise
    except Exception as exc:
        backend = "LangGraph" if settings.use_langgraph else "Nexi"
        logger.error("%s session start failed: %s", backend, exc)
        raise HTTPException(status_code=502, detail=f"{backend} unavailable")

    # Store conversation turn in working memory
    session_id = session_context["session_id"]
    await app.working_memory.append_turn(session_id, "user", body.raw_input)
    response_text = nexi_response.get("status", "") or nexi_response.get("error", "")
    await app.working_memory.append_turn(session_id, "assistant", response_text)

    return nexi_response


@router.post("/{session_id}/clarify")
async def clarify(session_id: str, body: ClarifyRequest, request: Request) -> dict[str, Any]:
    """Actor submits clarified input for a WAITING session."""
    app = request.app.state

    # Find session context by session_id (scan KV — v0: linear search)
    redis = app.kv_cache.redis_client
    cursor = 0
    session_context = None
    while True:
        cursor, keys = await redis.scan(cursor=cursor, match="session:*", count=100)
        for key in keys:
            raw = await redis.get(key)
            if raw:
                ctx = json.loads(raw)
                if ctx.get("session_id") == session_id:
                    session_context = ctx
                    break
        if cursor == 0 or session_context is not None:
            break

    if session_context is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    # Re-forward to Nexi with amended input, keeping same session_id
    session_context["raw_input"] = body.amended_input

    try:
        if settings.use_langgraph:
            nexi_response = await _run_langgraph_session(
                app,
                session_context,
                session_context.get("trace_id", session_id),
            )
        else:
            nexi_response = await _forward_to_nexi(session_context)
    except HTTPException:
        raise
    except Exception as exc:
        backend = "LangGraph" if settings.use_langgraph else "Nexi"
        logger.error("%s clarify failed: %s", backend, exc)
        raise HTTPException(status_code=502, detail=f"{backend} unavailable")

    # Append conversation turn
    await app.working_memory.append_turn(session_id, "user", body.amended_input)
    response_text = nexi_response.get("status", "") or nexi_response.get("error", "")
    await app.working_memory.append_turn(session_id, "assistant", response_text)

    return nexi_response
