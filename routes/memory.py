"""Steps 4 & 14: memory/read and memory/write."""
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from xnch.security.actor_sandbox import get_capabilities

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryReadRequest(BaseModel):
    session_id: str
    actor_id: str
    actor_role: str
    query: dict[str, Any]


class MemoryWriteRequest(BaseModel):
    session_id: str
    actor_id: str
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

    episodes = await app.episodic.fetch_for_manifest(
        intent_class=intent_class,
        entity_class=entity_class,
        actor_role=actor_role,
        lookback_days=lookback_days,
        max_episodes=max_episodes,
    )

    patterns = await app.pattern_store.fetch_for_manifest(
        intent_class=intent_class,
        entity_class=entity_class,
        actor_role=actor_role,
        max_patterns=max_patterns,
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

        await app.episodic.write_prediction_update(episode_id, prediction_delta, early_flag)

        app.event_log.emit(
            body.session_id, "xnch.memory", "PREDICTION_UPDATE_WRITTEN",
            data={"episode_id": episode_id, "prediction_delta": prediction_delta,
                  "early_flag": early_flag},
        )

        if early_flag:
            import asyncio
            asyncio.create_task(app.pattern_extractor.run_early())

        return {"status": "ok", "episode_id": episode_id}

    raise HTTPException(status_code=400, detail=f"Unknown write_type: {body.write_type}")


def _format_episode(ep: dict) -> dict:
    return {
        "episode_id": ep.get("episode_id"),
        "action_type": ep.get("action_type"),
        "entity_class": ep.get("entity_class"),
        "outcome": ep.get("outcome"),
        "created_at": _unix_to_iso(ep.get("created_at")),
    }


def _format_pattern(p: dict) -> dict:
    return {
        "pattern_id": p.get("pattern_id"),
        "context_signature": p.get("context_signature"),
        "success_rate": p.get("success_rate"),
        "confidence": p.get("confidence"),
        "observation_count": p.get("observation_count"),
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
