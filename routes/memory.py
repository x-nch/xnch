"""Steps 4 & 14: memory/read and memory/write."""
import asyncio
import json
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from xnch.security.actor_sandbox import get_capabilities

router = APIRouter(prefix="/memory", tags=["memory"])


class GraphEntityResponse(BaseModel):
    entity_id: str
    name: str
    type: str
    created_at: str | None = None


class GraphRelationResponse(BaseModel):
    from_id: str
    from_name: str | None = None
    to_id: str
    to_name: str | None = None
    rel_type: str
    confidence: float
    created_at: str | None = None


class GraphEntitiesPage(BaseModel):
    entities: list[GraphEntityResponse]
    total: int
    limit: int
    offset: int


class GraphRelationsPage(BaseModel):
    relations: list[GraphRelationResponse]
    total: int
    limit: int
    offset: int


class GraphSubgraphResponse(BaseModel):
    center_id: str
    depth: int
    entities: list[GraphEntityResponse]
    relations: list[GraphRelationResponse]


class GraphStatsResponse(BaseModel):
    entity_count: int
    relation_count: int
    types: dict[str, int] = Field(default_factory=dict)


class MemoryReadRequest(BaseModel):
    session_id: str
    actor_id: str
    actor_role: str
    query: dict[str, Any]


class MemoryWriteRequest(BaseModel):
    session_id: str
    actor_id: str
    actor_role: str | None = None
    write_type: str
    payload: dict[str, Any]


@router.post("/read")
async def memory_read(body: MemoryReadRequest, request: Request) -> dict[str, Any]:
    """Step 4: return context manifest — episodes, patterns, policies."""
    app = request.app.state
    q = body.query

    intent_class = q.get("intent_class", "")
    entity_class = q.get("target_entity_class", "")
    actor_role = body.actor_role
    lookback_days = q.get("lookback_window_days", 30)
    max_episodes = q.get("max_episodes", 20)
    max_patterns = q.get("max_patterns", 10)

    episodes = await app.pg_episodic.fetch_for_manifest(
        intent_class=intent_class,
        entity_class=entity_class,
        actor_role=actor_role,
        lookback_days=lookback_days,
        max_episodes=max_episodes,
    )

    patterns = await app.pg_episodic.fetch_patterns_for_manifest(
        intent_class=intent_class,
        entity_class=entity_class,
        actor_role=actor_role,
        max_patterns=max_patterns,
    )

    experiences = await app.experience_store.fetch_for_manifest(
        intent_class=intent_class,
        entity_class=entity_class,
        actor_role=actor_role,
        max_experiences=q.get("max_experiences", 10),
    )

    # Policies scoped to this context tuple
    policy_refs = _build_policy_refs(app, intent_class, entity_class, actor_role)

    state_version = await app.get_state_version()

    return {
        "manifest_id": str(uuid4()),
        "session_id": body.session_id,
        "system_state_version": state_version,
        "pinned_at": datetime.now(timezone.utc).isoformat(),
        "episodes": [_format_episode(ep) for ep in episodes],
        "patterns": [_format_pattern(p) for p in patterns],
        "experiences": [_format_experience(exp) for exp in experiences],
        "policies": policy_refs,
    }


@router.post("/write")
async def memory_write(body: MemoryWriteRequest, request: Request) -> dict[str, Any]:
    """Step 14: write prediction delta + early extraction flag to episode."""
    app = request.app.state

    caps = get_capabilities(body.actor_role)
    if not caps.can_write_memory:
        raise HTTPException(
            status_code=403,
            detail=f"Actor '{body.actor_role}' does not have write memory capability",
        )

    if body.write_type == "EPISODE_PREDICTION_UPDATE":
        payload = body.payload
        episode_id = payload.get("episode_id")
        prediction_delta = payload.get("prediction_delta")
        early_flag = payload.get("early_reextraction_flag", False)

        if not episode_id:
            raise HTTPException(status_code=422, detail="episode_id required")

        await app.pg_episodic.write_prediction_update(episode_id, prediction_delta, early_flag)

        app.event_log.emit(
            body.session_id, "xnch.memory", "PREDICTION_UPDATE_WRITTEN",
            data={"episode_id": episode_id, "prediction_delta": prediction_delta,
                  "early_flag": early_flag},
        )

        if early_flag:
            import asyncio
            asyncio.create_task(app.pattern_extractor.run())

        return {"status": "ok", "episode_id": episode_id}

    if body.write_type == "EXPERIENCE_REFLECTION":
        payload = body.payload
        required = ["context_signature", "intent_class", "action_type", "entity_class",
                    "actor_role", "outcome", "lesson", "insight", "verdict", "applicability"]
        missing = [f for f in required if not payload.get(f)]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing fields: {', '.join(missing)}")

        await app.experience_store.upsert_experience(
            context_signature=payload["context_signature"],
            intent_class=payload["intent_class"],
            action_type=payload["action_type"],
            entity_class=payload["entity_class"],
            actor_role=payload["actor_role"],
            outcome=payload["outcome"],
            lesson=payload["lesson"],
            insight=payload["insight"],
            verdict=payload["verdict"],
            applicability=payload["applicability"],
        )

        app.event_log.emit(
            body.session_id, "xnch.memory", "EXPERIENCE_REFLECTION_WRITTEN",
            data={"context_signature": payload["context_signature"],
                  "verdict": payload["verdict"]},
        )

        return {"status": "ok"}

    raise HTTPException(status_code=400, detail=f"Unknown write_type: {body.write_type}")


@router.get("/graph/stats", response_model=GraphStatsResponse)
async def graph_stats(request: Request) -> GraphStatsResponse:
    """Kuzu L3 graph summary — entity/relation counts and type distribution."""
    stats = request.app.state.graph_store.get_stats()
    return GraphStatsResponse(**stats)


@router.get("/graph/entities", response_model=GraphEntitiesPage)
async def graph_entities(
    request: Request,
    type: str | None = Query(default=None, description="Filter by entity type"),
    search: str | None = Query(default=None, description="Substring match on name"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> GraphEntitiesPage:
    store = request.app.state.graph_store
    entities = store.list_entities(
        type_filter=type,
        search=search,
        limit=limit,
        offset=offset,
    )
    total = store.count_entities(type_filter=type, search=search)
    return GraphEntitiesPage(
        entities=[GraphEntityResponse(**e) for e in entities],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/graph/relations", response_model=GraphRelationsPage)
async def graph_relations(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> GraphRelationsPage:
    store = request.app.state.graph_store
    relations = store.list_relations(limit=limit, offset=offset)
    total = store.count_relations()
    return GraphRelationsPage(
        relations=[GraphRelationResponse(**r) for r in relations],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/graph/subgraph", response_model=GraphSubgraphResponse)
async def graph_subgraph(
    request: Request,
    entity_id: str = Query(..., min_length=1),
    depth: int = Query(default=1, ge=1, le=2),
) -> GraphSubgraphResponse:
    store = request.app.state.graph_store
    raw = store.get_subgraph(entity_id=entity_id, depth=depth)
    return GraphSubgraphResponse(
        center_id=raw["center_id"],
        depth=raw["depth"],
        entities=[GraphEntityResponse(**e) for e in raw["entities"]],
        relations=[GraphRelationResponse(**r) for r in raw["relations"]],
    )


@router.get("/graph/stream")
async def graph_stream(request: Request) -> StreamingResponse:
    """SSE stream of Kuzu graph mutations and stats updates."""
    app = request.app.state
    store = app.graph_store
    broadcaster = app.graph_broadcaster

    async def event_stream():
        stats = store.get_stats()
        yield _sse({"type": "stats", **stats})
        yield _sse({"type": "ready"})

        queue = await broadcaster.subscribe()
        last_stats = stats
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=2.5)
                    yield _sse(event)
                    if event.get("type") == "stats":
                        last_stats = event
                except asyncio.TimeoutError:
                    current = store.get_stats()
                    if (
                        current["entity_count"] != last_stats.get("entity_count")
                        or current["relation_count"] != last_stats.get("relation_count")
                    ):
                        last_stats = current
                        yield _sse({"type": "stats", **current})
                        yield _sse({"type": "sync"})
                    yield _sse({"type": "heartbeat"})
        finally:
            broadcaster.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _format_episode(ep: dict) -> dict:
    duration_ms = None
    created = ep.get("created_at")
    completed = ep.get("completed_at")
    if created and completed:
        duration_ms = int((float(completed) - float(created)) * 1000)
    return {
        "episode_id": ep.get("episode_id"),
        "action_type": ep.get("action_type"),
        "entity_class": ep.get("entity_class"),
        "outcome": ep.get("outcome"),
        "duration_ms": duration_ms,
        "created_at": _unix_to_iso(created),
    }


def _format_pattern(p: dict) -> dict:
    return {
        "pattern_id": p.get("pattern_id"),
        "context_signature": p.get("context_signature"),
        "success_rate": p.get("success_rate"),
        "confidence": p.get("confidence"),
        "observation_count": p.get("observation_count"),
    }


def _format_experience(e: dict) -> dict:
    return {
        "experience_id": e.get("experience_id"),
        "context_signature": e.get("context_signature"),
        "intent_class": e.get("intent_class"),
        "action_type": e.get("action_type"),
        "entity_class": e.get("entity_class"),
        "actor_role": e.get("actor_role"),
        "outcome": e.get("outcome"),
        "lesson": e.get("lesson"),
        "insight": e.get("insight"),
        "verdict": e.get("verdict"),
        "applicability": e.get("applicability"),
        "confidence": e.get("confidence"),
        "created_at": _unix_to_iso(e.get("created_at")),
    }


def _build_policy_refs(app, intent_class: str, entity_class: str, actor_role: str) -> list[dict]:
    refs = []
    for rule in app.policy_engine._rules:
        c = rule.conditions
        intent_match = not c.intent_class or c.intent_class == intent_class
        entity_match = not c.entity_class or c.entity_class == entity_class
        role_match = not c.actor_role or c.actor_role == actor_role
        if intent_match and entity_match and role_match:
            refs.append({
                "policy_id": rule.rule_id,
                "rule_expression": f"{c.intent_class or '*'}|{c.action_type or '*'}|{c.entity_class or '*'}|{c.actor_role or '*'}",
                "enforcement_level": rule.action.verdict,
            })
    return refs


def _unix_to_iso(ts) -> str | None:
    if ts is None:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
