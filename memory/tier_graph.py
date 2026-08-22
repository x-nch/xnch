"""Unified 4-tier memory graph assembler.

Reads live from L0 SensoryBuffer (Redis), L1 WorkingMemory (Redis),
L2 PgEpisodicStore (Postgres), and L3 GraphStore (Kuzu) and flattens them
into a single node/edge model tagged with a `tier` label. Powers the web
memory graph explorer endpoints:

- GET /memory/graph/tiers  → per-tier node/edge counts
- GET /memory/graph/all    → unified paginated nodes + edges

Cross-tier edges:
- session (L1) ──produced──▶ episode (L2)      via episodes.session_id
- episode (L2) ──mentions──▶ entity (L3)       heuristic name match in raw_text

Every store access is fail-open: a disconnected store contributes an empty
tier instead of raising.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SENSORY_LIMIT = 50
SESSION_LIMIT = 20
TURNS_PER_SESSION = 20
EPISODE_LIMIT = 200
EPISODE_HOURS = 24 * 30
SEMANTIC_LIMIT = 400
MAX_MENTION_EDGES_PER_EPISODE = 8

TIER_IDS = ("sensory", "working", "episodic", "semantic")


def _resolve_tiers(tier: str | None) -> tuple[str, ...]:
    """Parse a comma-separated tier filter; empty/invalid → all tiers."""
    if not tier:
        return TIER_IDS
    parts = [t.strip() for t in tier.split(",") if t.strip()]
    valid = tuple(t for t in parts if t in TIER_IDS)
    return valid if valid else TIER_IDS


# ---------------------------------------------------------------------- #
# L0 — Sensory (Redis)
# ---------------------------------------------------------------------- #

def _sensory_node(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source", "unknown")
    data = str(payload.get("data", ""))
    ts = payload.get("timestamp", 0.0)
    return {
        "id": f"sensory:{source}:{int(float(ts) * 1000)}",
        "name": source,
        "type": source,
        "tier": "sensory",
        "created_at": None,
        "detail": data[:400],
    }


async def _sensory_nodes(
    sensory_buffer: Any, limit: int = SENSORY_LIMIT
) -> list[dict[str, Any]]:
    if sensory_buffer is None:
        return []
    try:
        perceptions = await sensory_buffer.read_recent_all(limit=limit)
    except Exception:
        logger.debug("sensory buffer read failed", exc_info=True)
        return []
    return [_sensory_node(p) for p in perceptions]


# ---------------------------------------------------------------------- #
# L1 — Working memory (Redis)
# ---------------------------------------------------------------------- #

def _session_node(session_id: str) -> dict[str, Any]:
    return {
        "id": f"working:{session_id}",
        "name": f"session {session_id[:8]}",
        "type": "session",
        "tier": "working",
        "created_at": None,
        "detail": None,
    }


def _turn_node(session_id: str, idx: int, turn: dict[str, Any]) -> dict[str, Any]:
    role = turn.get("role", "turn")
    content = str(turn.get("content", ""))
    return {
        "id": f"working:{session_id}:{idx}",
        "name": f"{role}: {content[:40]}",
        "type": "turn",
        "tier": "working",
        "created_at": None,
        "detail": content[:400],
    }


async def _working_graph(
    working_memory: Any,
    session_limit: int = SESSION_LIMIT,
    turns: int = TURNS_PER_SESSION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    if working_memory is None:
        return [], [], set()
    try:
        sessions = await working_memory.list_sessions(limit=session_limit)
    except Exception:
        logger.debug("working memory read failed", exc_info=True)
        return [], [], set()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    session_ids: set[str] = set()
    for s in sessions:
        sid = s.get("session_id")
        if not sid:
            continue
        session_ids.add(sid)
        nodes.append(_session_node(sid))
        try:
            turn_list = await working_memory.get_turns(sid, last_n=turns)
        except Exception:
            turn_list = []
        for idx, turn in enumerate(turn_list):
            nodes.append(_turn_node(sid, idx, turn))
            edges.append({
                "from_id": f"working:{sid}",
                "to_id": f"working:{sid}:{idx}",
                "rel_type": "has_turn",
                "confidence": 1.0,
                "tier": "working",
            })
    return nodes, edges, session_ids


# ---------------------------------------------------------------------- #
# L2 — Episodic (Postgres)
# ---------------------------------------------------------------------- #

def _episode_node(ep: dict[str, Any]) -> dict[str, Any]:
    summary = (ep.get("summary") or "").strip()
    raw = (ep.get("raw_text") or "").strip()
    name = summary or raw[:60]
    return {
        "id": f"episode:{ep['id']}",
        "name": name[:80],
        "type": ep.get("type", "episode"),
        "tier": "episodic",
        "created_at": ep.get("timestamp"),
        "detail": raw[:500] or None,
    }


async def _episodic_graph(
    pg_episodic: Any,
    limit: int = EPISODE_LIMIT,
    hours: int = EPISODE_HOURS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if pg_episodic is None:
        return [], []
    try:
        episodes = await pg_episodic.list_recent(hours=hours, limit=limit)
    except Exception:
        logger.debug("episodic store read failed", exc_info=True)
        return [], []
    nodes = [_episode_node(ep) for ep in episodes]
    meta = [
        {
            "id": f"episode:{ep['id']}",
            "session_id": ep.get("session_id"),
            "raw_text": ep.get("raw_text") or "",
        }
        for ep in episodes
    ]
    return nodes, meta


# ---------------------------------------------------------------------- #
# L3 — Semantic (Kuzu)
# ---------------------------------------------------------------------- #

def _semantic_node(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"semantic:{e['entity_id']}",
        "name": e["name"],
        "type": e["type"],
        "tier": "semantic",
        "created_at": e.get("created_at"),
        "detail": None,
    }


async def _semantic_graph(
    graph_store: Any, limit: int = SEMANTIC_LIMIT
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if graph_store is None:
        return [], []
    try:
        entities = graph_store.list_entities(limit=limit)
        relations = graph_store.list_relations(limit=limit)
    except Exception:
        logger.debug("semantic graph read failed", exc_info=True)
        return [], []
    nodes = [_semantic_node(e) for e in entities]
    edges = [
        {
            "from_id": f"semantic:{r['from_id']}",
            "to_id": f"semantic:{r['to_id']}",
            "rel_type": r["rel_type"],
            "confidence": float(r["confidence"]),
            "tier": "semantic",
        }
        for r in relations
    ]
    return nodes, edges


# ---------------------------------------------------------------------- #
# Cross-tier edges
# ---------------------------------------------------------------------- #

def _session_edges(
    working_nodes: list[dict[str, Any]], episode_meta: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Link visible working sessions to episodes they produced."""
    session_nodes = {n["id"] for n in working_nodes if n["type"] == "session"}
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ep in episode_meta:
        sid = ep.get("session_id")
        if not sid:
            continue
        from_id = f"working:{sid}"
        to_id = ep["id"]
        key = (from_id, to_id)
        if from_id in session_nodes and key not in seen:
            seen.add(key)
            edges.append({
                "from_id": from_id,
                "to_id": to_id,
                "rel_type": "produced",
                "confidence": 1.0,
                "tier": "cross",
            })
    return edges


def _mentions_edges(
    episode_meta: list[dict[str, Any]], semantic_nodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Link episodes to entities whose name appears in the raw text."""
    name_map: dict[str, str] = {}
    for n in semantic_nodes:
        nm = (n.get("name") or "").strip().lower()
        if nm and nm not in name_map:
            name_map[nm] = n["id"]
    if not name_map:
        return []
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ep in episode_meta:
        text = (ep.get("raw_text") or "").lower()
        if not text:
            continue
        count = 0
        for nm, eid in name_map.items():
            if nm in text:
                key = (ep["id"], eid)
                if key in seen:
                    continue
                seen.add(key)
                edges.append({
                    "from_id": ep["id"],
                    "to_id": eid,
                    "rel_type": "mentions",
                    "confidence": 0.6,
                    "tier": "cross",
                })
                count += 1
                if count >= MAX_MENTION_EDGES_PER_EPISODE:
                    break
    return edges


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #

def _stores(stores: Any) -> tuple[Any, Any, Any, Any]:
    return (
        getattr(stores, "sensory_buffer", None),
        getattr(stores, "working_memory", None),
        getattr(stores, "pg_episodic", None),
        getattr(stores, "graph_store", None),
    )


async def assemble_tier_graph(
    stores: Any,
    *,
    tier: str | None = None,
    search: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> dict[str, Any]:
    """Build the unified node/edge view, filtered and paginated."""
    sensory_buffer, working_memory, pg_episodic, graph_store = _stores(stores)
    include = set(_resolve_tiers(tier))

    sensory_nodes = (
        await _sensory_nodes(sensory_buffer) if "sensory" in include else []
    )
    working_nodes, working_edges, _ = (
        await _working_graph(working_memory) if "working" in include else ([], [], set())
    )
    episodic_nodes, episode_meta = (
        await _episodic_graph(pg_episodic) if "episodic" in include else ([], [])
    )
    semantic_nodes, semantic_edges = (
        await _semantic_graph(graph_store) if "semantic" in include else ([], [])
    )

    edges: list[dict[str, Any]] = list(working_edges) + list(semantic_edges)
    if {"working", "episodic"} <= include:
        edges += _session_edges(working_nodes, episode_meta)
    if {"episodic", "semantic"} <= include:
        edges += _mentions_edges(episode_meta, semantic_nodes)

    nodes = sensory_nodes + working_nodes + episodic_nodes + semantic_nodes

    if search:
        q = search.lower()
        nodes = [
            n for n in nodes
            if q in (n["name"] or "").lower()
            or (n.get("detail") or "").lower().find(q) >= 0
        ]
        kept = {n["id"] for n in nodes}
        edges = [e for e in edges if e["from_id"] in kept and e["to_id"] in kept]

    total = len(nodes)
    paged_nodes = nodes[offset:offset + limit]
    kept_ids = {n["id"] for n in paged_nodes}
    paged_edges = [e for e in edges if e["from_id"] in kept_ids and e["to_id"] in kept_ids]

    counts = {t: 0 for t in TIER_IDS}
    for n in nodes:
        counts[n["tier"]] += 1

    return {
        "nodes": paged_nodes,
        "edges": paged_edges,
        "total": total,
        "tiers": counts,
        "limit": limit,
        "offset": offset,
    }


async def tier_summary(stores: Any) -> dict[str, Any]:
    """Per-tier node/edge counts plus cross-tier edge count."""
    sensory_buffer, working_memory, pg_episodic, graph_store = _stores(stores)

    sensory_nodes = await _sensory_nodes(sensory_buffer)
    working_nodes, working_edges, _ = await _working_graph(working_memory)
    episodic_nodes, episode_meta = await _episodic_graph(pg_episodic)
    semantic_nodes, semantic_edges = await _semantic_graph(graph_store)

    cross_edges = len(_session_edges(working_nodes, episode_meta)) + len(
        _mentions_edges(episode_meta, semantic_nodes)
    )

    return {
        "tiers": {
            "sensory": {"nodes": len(sensory_nodes), "edges": 0},
            "working": {"nodes": len(working_nodes), "edges": len(working_edges)},
            "episodic": {"nodes": len(episodic_nodes), "edges": 0},
            "semantic": {"nodes": len(semantic_nodes), "edges": len(semantic_edges)},
        },
        "cross_edges": cross_edges,
    }
