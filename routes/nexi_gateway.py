import json
import logging
import os
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from nexi.character.prompt_loader import build_system_prompt, load_capabilities
from nexi.pipeline.context_assembler import assemble_context
from nexi.proactivity.engine import ProactivityEngine
from xnch.config import settings
from xnch.routing.response_sanitize import strip_thinking
from xnch.security.injection_guard import scan_input
from xnch.security.memory_guard import validate_memory_write
from xnch.security.trust_model import get_trust_level

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/nexi", tags=["nexi"])

SYSTEM_PROMPT_CACHE_KEY = "nexi:system-prompt"
SYSTEM_PROMPT_CACHE_TTL = 60

from xnch.routing.recall_intent import recall_query as _recall_query

OPENCODE_GO_BASE = os.environ.get("OPENCODE_GO_BASE_URL", settings.opencode_go_api_url)
OPENCODE_GO_API_KEY = os.environ.get("OPENCODE_GO_API_KEY", settings.opencode_go_api_key)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    actor_role: str = "operator"


class MemoryRecallRequest(BaseModel):
    query: str
    top_k: int = 5


def _get_proactivity(app) -> ProactivityEngine:
    if not hasattr(app, "_nexi_proactivity"):
        redis = app.kv_cache.redis_client
        app._nexi_proactivity = ProactivityEngine(redis)
    return app._nexi_proactivity


async def _agent_lessons_for_chat(app: Any, message: str) -> list[str]:
    if not settings.am_prefetch_enabled:
        return []
    from xnch.memory.agentmemory_prefetch import prefetch_agent_lessons

    query = _recall_query(message) or message
    return await prefetch_agent_lessons(app, query)


async def _safe_redis_delete(redis, key: str) -> None:
    try:
        await redis.delete(key)
    except Exception as exc:
        logger.warning("Redis delete failed for key %s: %s", key, exc)


def _invalidate_system_prompt_cache(app) -> None:
    redis = app.kv_cache.redis_client
    import asyncio
    asyncio.ensure_future(_safe_redis_delete(redis, SYSTEM_PROMPT_CACHE_KEY))


@router.get("/system-prompt", response_class=PlainTextResponse)
async def get_system_prompt(request: Request) -> str:
    app = request.app.state
    redis = app.kv_cache.redis_client

    cached = await redis.get(SYSTEM_PROMPT_CACHE_KEY)
    if cached:
        return cached

    entities = app.graph_store.fetch_entities(limit=20) if hasattr(app, "graph_store") else []
    recent_entities = [e.get("document", "") for e in entities if e.get("document")]

    prompt = build_system_prompt(
        session_memory=[], recent_entities=recent_entities, include_capabilities=True
    )
    await redis.set(SYSTEM_PROMPT_CACHE_KEY, prompt, ex=SYSTEM_PROMPT_CACHE_TTL)
    return prompt


@router.get("/capabilities")
async def get_capabilities() -> dict[str, Any]:
    return load_capabilities()


@router.get("/tools")
async def get_tools() -> dict[str, Any]:
    """Live tool inventory for the nexi actor (native + bridged), read-only."""
    from xnch_mcp.bridge.pool import get_bridge_pool
    from xnch_mcp.registry import list_openai_tools

    tools = list_openai_tools("nexi")
    pool = get_bridge_pool()
    servers = pool.server_status() if pool is not None else []
    return {
        "tools": tools,
        "bridge": {
            "active": bool(pool is not None and pool.started),
            "servers": servers,
        },
    }


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> dict[str, Any]:
    app = request.app.state

    result = scan_input(body.message, app.event_log)
    if not result.is_clean:
        raise HTTPException(status_code=400, detail="Input rejected by injection guard")

    ctx = await assemble_context(
        session_id=body.session_id,
        raw_input=body.message,
        working_memory=app.working_memory,
        pg_episodic=app.pg_episodic,
        graph_store=app.graph_store,
        relationship_store=app.relationship_store,
        sensory_buffer=app.sensory_buffer,
        proactivity_engine=_get_proactivity(app),
        recall_query=_recall_query(body.message),
        agent_lessons=await _agent_lessons_for_chat(app, body.message),
    )

    messages = ctx.to_messages(body.message)
    model_name = settings.llm_model_id

    await app.working_memory.append_turn(body.session_id, "user", body.message)

    from xnch_mcp.chat_tools import chat_with_tools

    try:
        response_text = await chat_with_tools(
            app,
            messages,
            model_name,
            session_id=body.session_id,
            actor_role="nexi",
        )
    except Exception as exc:
        logger.error("LiteLLM call failed: %s", exc)
        raise HTTPException(status_code=502, detail="LiteLLM unavailable")

    await app.working_memory.append_turn(body.session_id, "assistant", response_text)

    episode_text = f"{body.message}\n{response_text}"
    validation = validate_memory_write(
        content=episode_text,
        actor_role=body.actor_role,
        trust_level=get_trust_level(body.actor_role),
    )
    if not validation[0]:
        logger.warning("Memory write blocked by guard: %s", validation[1])
    elif await app.pg_episodic.has_identical_recent(episode_text, hours=24):
        logger.info("Skipping duplicate episode store for session %s", body.session_id)
    else:
        await app.pg_episodic.store_episode(
            type_="conversation",
            raw_text=episode_text,
            summary=f"{body.message[:80]} → {response_text[:120]}",
        )

    _invalidate_system_prompt_cache(app)

    return {
        "response": response_text,
        "model_used": model_name,
        "session_id": body.session_id,
    }


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    app = request.app.state

    result = scan_input(body.message, app.event_log)
    if not result.is_clean:
        raise HTTPException(status_code=400, detail="Input rejected by injection guard")

    ctx = await assemble_context(
        session_id=body.session_id,
        raw_input=body.message,
        working_memory=app.working_memory,
        pg_episodic=app.pg_episodic,
        graph_store=app.graph_store,
        relationship_store=app.relationship_store,
        sensory_buffer=app.sensory_buffer,
        proactivity_engine=_get_proactivity(app),
        recall_query=_recall_query(body.message),
        agent_lessons=await _agent_lessons_for_chat(app, body.message),
    )

    messages = ctx.to_messages(body.message)
    model_name = settings.llm_model_id

    await app.working_memory.append_turn(body.session_id, "user", body.message)

    async def event_stream():
        from xnch_mcp.chat_tools import chat_with_tools

        try:
            full_text = await chat_with_tools(
                app,
                messages,
                model_name,
                session_id=body.session_id,
                actor_role="nexi",
            )
        except Exception as exc:
            logger.error("LiteLLM stream/tool loop failed: %s", exc)
            yield f"data: {json.dumps({'error': 'LiteLLM unavailable'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        if full_text:
            yield f"data: {json.dumps({'content': full_text})}\n\n"

        await app.working_memory.append_turn(body.session_id, "assistant", full_text)
        episode_text = f"{body.message}\n{full_text}"
        validation = validate_memory_write(
            content=episode_text,
            actor_role=body.actor_role,
            trust_level=get_trust_level(body.actor_role),
        )
        if validation[0] and await app.pg_episodic.has_identical_recent(episode_text, hours=24):
            logger.info("Skipping duplicate episode store for session %s", body.session_id)
        elif validation[0]:
            await app.pg_episodic.store_episode(
                type_="conversation",
                raw_text=episode_text,
                summary=f"{body.message[:80]} → {full_text[:120]}",
            )
        else:
            logger.warning("Memory write blocked by guard: %s", validation[1])
        _invalidate_system_prompt_cache(app)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/memory/surface")
async def memory_surface(request: Request) -> list[dict[str, Any]]:
    app = request.app.state
    proactivity = _get_proactivity(app)
    events = await proactivity.get_pending()
    return [e.to_dict() for e in events]


@router.post("/memory/recall")
async def memory_recall(body: MemoryRecallRequest, request: Request) -> list[dict[str, Any]]:
    app = request.app.state

    episodes = await app.pg_episodic.retrieve_similar(
        query_text=body.query, top_k=body.top_k
    )

    results: list[dict[str, Any]] = []
    for ep in episodes:
        result: dict[str, Any] = {
            "id": ep.get("id"),
            "type": ep.get("type", "episode"),
            "timestamp": ep.get("timestamp"),
            "content": ep.get("raw_text") or ep.get("summary", ""),
            "similarity": ep.get("similarity", 0.0),
            "importance": ep.get("importance", 0.0),
        }

        entity_id = ""
        text = ep.get("raw_text") or ep.get("summary", "")
        if text:
            entity_node = app.graph_store.get_entity_by_name(text[:50])
            entity_id = entity_node["metadata"].get("entity_id", "") if entity_node else ""
        if entity_id:
            try:
                rels = await app.relationship_store.get_relationships(entity_id)
                if rels:
                    result["relationships"] = [
                        {
                            "entity_a": r.entity_a_id,
                            "entity_b": r.entity_b_id,
                            "type": r.relationship_type,
                            "strength": r.strength,
                        }
                        for r in rels
                    ]
            except Exception:
                pass

        results.append(result)

    return results
