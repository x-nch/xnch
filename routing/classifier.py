"""LiteLLM routing classifier with a Redis exact-match cache.

Replaces the agentmemory/ChromaDB routing-decisions cache. Decisions are
keyed on the normalized raw input so repeated requests are served from
Redis without an LLM lookup.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import redis

from xnch.config import settings

logger = logging.getLogger(__name__)

# Decision cache, not durable memory: Redis is the short-term tier by design
# (sensory/working; PG episodic + Kuzu graph are the durable tiers). A miss
# re-runs the deterministic routing logic (no LLM call), so expiry is lossless
# — this never had indefinite-retention semantics under agentmemory either.
_ROUTING_CACHE_TTL_S = 7 * 86400
_cache: redis.Redis | None = None


def _get_redis() -> redis.Redis | None:
    global _cache
    if _cache is None:
        try:
            _cache = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        except Exception as exc:
            logger.warning("Routing cache unavailable: %s", exc)
            _cache = None
    return _cache


def _cache_key(raw_input: str) -> str:
    digest = hashlib.sha256(raw_input.lower().strip().encode("utf-8")).hexdigest()
    return f"xnch:routing:{digest}"


def _cache_lookup(raw_input: str) -> ModelRoute | None:
    client = _get_redis()
    if client is None:
        return None
    try:
        raw = client.get(_cache_key(raw_input))
        if not raw:
            return None
        parsed = json.loads(raw)
        return ModelRoute(
            model_name=parsed["model_name"],
            reason=f"recalled: {parsed['reason']}",
        )
    except Exception as exc:
        logger.warning("Routing cache lookup failed: %s", exc)
        return None


def _cache_store(raw_input: str, route: ModelRoute, actor_role: str, metadata: dict) -> None:
    client = _get_redis()
    if client is None:
        return
    try:
        client.set(
            _cache_key(raw_input),
            json.dumps({
                "raw_input": raw_input,
                "actor_role": actor_role,
                "intent_class": metadata.get("intent_class", ""),
                "model_name": route.model_name,
                "reason": route.reason,
            }),
            ex=_ROUTING_CACHE_TTL_S,
        )
    except Exception as exc:
        logger.warning("Routing cache store failed: %s", exc)


def _compute_complexity(raw_input: str, metadata: dict) -> float:
    if 'complexity_score' in metadata:
        return float(metadata['complexity_score'])
    length_score = min(len(raw_input) / 500, 1.0)
    has_multipart = 1.0 if any(c in raw_input for c in ['and then', 'after that', 'first', 'finally', 'steps']) else 0.0
    has_code = 0.5 if any(c in raw_input for c in ['```', 'def ', 'class ', 'import ', 'kubectl']) else 0.0
    return min((length_score + has_multipart + has_code) / 3, 1.0)


@dataclass
class ModelRoute:
    model_name: str
    reason: str


def classify_request(raw_input: str, actor_role: str, metadata: dict) -> ModelRoute:
    cached = _cache_lookup(raw_input)
    if cached is not None:
        return cached

    if metadata.get("privacy_sensitive"):
        route = ModelRoute(
            model_name="qwen2.5-vl-7b",
            reason="privacy_sensitive: routed to local model",
        )
        _cache_store(raw_input, route, actor_role, metadata)
        return route

    intent_class = metadata.get("intent_class", "")
    complexity_score = metadata.get("complexity_score", 0.0)

    if intent_class == "EXECUTION":
        route = ModelRoute(
            model_name="qwen2.5-vl-7b",
            reason="intent_class=EXECUTION: routed to local model for low-latency execution",
        )
        _cache_store(raw_input, route, actor_role, metadata)
        return route

    if intent_class == "DECISION":
        complexity_score = _compute_complexity(raw_input, metadata)
        if complexity_score > 0.7:
            route = ModelRoute(
                model_name="qwen2.5-vl-7b",
                reason=f"intent_class=DECISION complexity={complexity_score:.2f}: routed to qwen-vl",
            )
            _cache_store(raw_input, route, actor_role, metadata)
            return route

    route = ModelRoute(
        model_name="qwen2.5-vl-7b",
        reason="default route: qwen-vl",
    )
    _cache_store(raw_input, route, actor_role, metadata)
    return route
