import json
import logging
import os
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from nexi.character.prompt_loader import build_system_prompt
from nexi.pipeline.context_assembler import assemble_context
from nexi.proactivity.engine import ProactivityEngine
from xnch.routing.classifier import classify_request
from xnch.security.injection_guard import scan_input
from xnch.security.memory_guard import validate_memory_write
from xnch.security.trust_model import get_trust_level

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/nexi", tags=["nexi"])

SYSTEM_PROMPT_CACHE_KEY = "nexi:system-prompt"
SYSTEM_PROMPT_CACHE_TTL = 60

LITELLM_BASE = os.environ.get("LITELLM_BASE_URL", "http://i7-node:4000")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    actor_role: str = "openclaw"


class MemoryRecallRequest(BaseModel):
    query: str
    top_k: int = 5


def _get_proactivity(app) -> ProactivityEngine:
    if not hasattr(app, "_nexi_proactivity"):
        redis = app.kv_cache.redis_client
        app._nexi_proactivity = ProactivityEngine(redis)
    return app._nexi_proactivity


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

    from agentmemory import get_memories
    entities = get_memories("entities", n_results=20)
    recent_entities = [e.get("document", "") for e in entities if e.get("document")]

    prompt = build_system_prompt(session_memory=[], recent_entities=recent_entities)
    await redis.set(SYSTEM_PROMPT_CACHE_KEY, prompt, ex=SYSTEM_PROMPT_CACHE_TTL)
    return prompt


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
    )

    route = classify_request(body.message, body.actor_role, {})
    messages = ctx.to_messages(body.message)
    model_name = route.model_name

    await app.working_memory.append_turn(body.session_id, "user", body.message)

    try:
        async with httpx.AsyncClient(base_url=LITELLM_BASE, timeout=120.0) as client:
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                },
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"} if LITELLM_API_KEY else {},
            )
            resp.raise_for_status()
            response_text = resp.json()["choices"][0]["message"]["content"]
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
    else:
        await app.pg_episodic.store_episode(
            type_="conversation",
            raw_text=episode_text,
            summary=f"OpenClaw chat: {body.message[:100]}",
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
    )

    route = classify_request(body.message, body.actor_role, {})
    messages = ctx.to_messages(body.message)
    model_name = route.model_name

    await app.working_memory.append_turn(body.session_id, "user", body.message)

    async def event_stream():
        full_text = ""
        try:
            async with httpx.AsyncClient(base_url=LITELLM_BASE, timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    "/chat/completions",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": 2048,
                        "temperature": 0.7,
                        "stream": True,
                    },
                    headers={"Authorization": f"Bearer {LITELLM_API_KEY}"} if LITELLM_API_KEY else {},
                ) as resp:
                    if resp.status_code != 200:
                        yield f"data: {json.dumps({'error': f'LiteLLM error {resp.status_code}'})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line.removeprefix("data: ")
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                full_text += delta
                                yield f"data: {json.dumps({'content': delta})}\n\n"
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.error("LiteLLM stream failed: %s", exc)
            yield f"data: {json.dumps({'error': 'LiteLLM unavailable'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        await app.working_memory.append_turn(body.session_id, "assistant", full_text)
        episode_text = f"{body.message}\n{full_text}"
        validation = validate_memory_write(
            content=episode_text,
            actor_role=body.actor_role,
            trust_level=get_trust_level(body.actor_role),
        )
        if validation[0]:
            await app.pg_episodic.store_episode(
                type_="conversation",
                raw_text=episode_text,
                summary=f"OpenClaw chat: {body.message[:100]}",
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
