"""Unit tests for the unified 4-tier memory graph assembler.

Covers node/edge construction per tier, cross-tier edge generation
(session→episode via session_id, episode→entity via mentions heuristic),
tier filtering, search filtering, pagination, and fail-open behaviour when
stores are missing or raise.
"""

from __future__ import annotations

from typing import Any

import pytest

from xnch.memory.tier_graph import (
    TIER_IDS,
    assemble_tier_graph,
    tier_summary,
)


class FakeSensoryBuffer:
    def __init__(self, perceptions: list[dict[str, Any]]) -> None:
        self._perceptions = perceptions

    async def read_recent_all(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._perceptions[:limit]


class FakeWorkingMemory:
    def __init__(self, sessions: dict[str, list[dict[str, Any]]]) -> None:
        self._sessions = sessions

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            {"session_id": sid, "turns": len(turns)}
            for sid, turns in list(self._sessions.items())[:limit]
        ]

    async def get_turns(
        self, session_id: str, last_n: int = 20
    ) -> list[dict[str, Any]]:
        return self._sessions.get(session_id, [])[-last_n:]


class FakeEpisodic:
    def __init__(self, episodes: list[dict[str, Any]]) -> None:
        self._episodes = episodes

    async def list_recent(self, hours: int = 24, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self._episodes
        if limit is not None:
            rows = rows[:limit]
        return rows


class FakeGraphStore:
    def __init__(self, entities: list[dict[str, Any]], relations: list[dict[str, Any]]) -> None:
        self._entities = entities
        self._relations = relations

    def list_entities(self, **_: Any) -> list[dict[str, Any]]:
        return self._entities

    def list_relations(self, **_: Any) -> list[dict[str, Any]]:
        return self._relations


class RaisingStore:
    async def read_recent_all(self, limit: int = 50) -> list[dict[str, Any]]:
        raise RuntimeError("boom")

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        raise RuntimeError("boom")

    async def get_turns(self, session_id: str, last_n: int = 20) -> list[dict[str, Any]]:
        raise RuntimeError("boom")

    async def list_recent(self, hours: int = 24, limit: int | None = None) -> list[dict[str, Any]]:
        raise RuntimeError("boom")

    def list_entities(self, **_: Any) -> list[dict[str, Any]]:
        raise RuntimeError("boom")

    def list_relations(self, **_: Any) -> list[dict[str, Any]]:
        raise RuntimeError("boom")


class _Stores:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _stores(**kwargs: Any) -> _Stores:
    return _Stores(**kwargs)


def _episode(session_id: str | None = None, raw: str = "refactored the xnch gateway") -> dict[str, Any]:
    return {
        "id": "uuid-1",
        "type": "execution",
        "raw_text": raw,
        "summary": "did a thing",
        "importance": 0.5,
        "recall_count": 0,
        "last_recalled": None,
        "timestamp": "2026-01-01T00:00:00Z",
        "decay_score": 0.0,
        "archived": False,
        "session_id": session_id,
    }


@pytest.fixture
def stores() -> _Stores:
    sensory = FakeSensoryBuffer(
        [{"source": "voice", "data": "hello memory graph", "timestamp": 100.0}]
    )
    working = FakeWorkingMemory(
        {"sess-123": [{"role": "user", "content": "what is xnch?"}]}
    )
    episodic = FakeEpisodic(
        [
            _episode(session_id="sess-123", raw="user asked about xnch"),
            _episode(session_id="sess-other", raw="kuzu storage notes"),
        ]
    )
    semantic = FakeGraphStore(
        entities=[
            {"entity_id": "ent-1", "name": "xnch", "type": "project", "created_at": None}
        ],
        relations=[
            {"from_id": "ent-1", "to_id": "ent-1", "rel_type": "self_ref", "confidence": 0.9, "created_at": None}
        ],
    )
    return _stores(
        sensory_buffer=sensory,
        working_memory=working,
        pg_episodic=episodic,
        graph_store=semantic,
    )


async def test_all_tiers_assembled(stores: _Stores) -> None:
    data = await assemble_tier_graph(stores)
    tiers = {n["tier"] for n in data["nodes"]}
    assert tiers == set(TIER_IDS)
    assert any(n["id"].startswith("sensory:") for n in data["nodes"])
    assert any(n["id"] == "working:sess-123" for n in data["nodes"])
    assert any(n["id"] == "episode:uuid-1" for n in data["nodes"])
    assert any(n["id"] == "semantic:ent-1" for n in data["nodes"])
    assert data["total"] == len(data["nodes"])


async def test_cross_tier_edges(stores: _Stores) -> None:
    data = await assemble_tier_graph(stores)
    rel_types = {(e["from_id"], e["to_id"], e["rel_type"], e["tier"]) for e in data["edges"]}
    assert ("working:sess-123", "episode:uuid-1", "produced", "cross") in rel_types
    assert ("episode:uuid-1", "semantic:ent-1", "mentions", "cross") in rel_types


async def test_produced_edge_requires_matching_session(stores: _Stores) -> None:
    data = await assemble_tier_graph(stores)
    bad = [
        e for e in data["edges"]
        if e["rel_type"] == "produced" and e["from_id"] == "working:sess-other"
    ]
    assert bad == []


async def test_tier_filter(stores: _Stores) -> None:
    data = await assemble_tier_graph(stores, tier="episodic,semantic")
    assert {n["tier"] for n in data["nodes"]} == {"episodic", "semantic"}
    produced = [e for e in data["edges"] if e["rel_type"] == "produced"]
    assert produced == []
    assert any(e["rel_type"] == "mentions" for e in data["edges"])


async def test_invalid_tier_falls_back_to_all(stores: _Stores) -> None:
    data = await assemble_tier_graph(stores, tier="bogus")
    assert {n["tier"] for n in data["nodes"]} == set(TIER_IDS)


async def test_search_filters_nodes_and_edges(stores: _Stores) -> None:
    data = await assemble_tier_graph(stores, search="xnch")
    names = {n["name"] for n in data["nodes"]}
    assert any("xnch" in n.lower() for n in names)
    for e in data["edges"]:
        ids = {data["nodes"][i]["id"] for i in range(len(data["nodes"]))}
        assert e["from_id"] in ids and e["to_id"] in ids


async def test_pagination(stores: _Stores) -> None:
    data = await assemble_tier_graph(stores, limit=2, offset=0)
    assert len(data["nodes"]) == 2
    assert data["limit"] == 2
    assert data["offset"] == 0
    assert data["total"] > 2


async def test_tier_summary(stores: _Stores) -> None:
    summary = await tier_summary(stores)
    assert summary["tiers"]["sensory"]["nodes"] == 1
    assert summary["tiers"]["working"]["nodes"] == 2
    assert summary["tiers"]["working"]["edges"] == 1
    assert summary["tiers"]["episodic"]["nodes"] == 2
    assert summary["tiers"]["semantic"]["nodes"] == 1
    assert summary["cross_edges"] == 2


async def test_fail_open_when_stores_missing() -> None:
    data = await assemble_tier_graph(_stores())
    assert data["nodes"] == []
    assert data["edges"] == []

    summary = await tier_summary(_stores())
    assert summary["cross_edges"] == 0


async def test_fail_open_when_stores_raise() -> None:
    bad = _stores(sensory_buffer=RaisingStore(), working_memory=RaisingStore(), pg_episodic=RaisingStore(), graph_store=RaisingStore())
    data = await assemble_tier_graph(bad)
    assert data["nodes"] == []
    assert data["edges"] == []
