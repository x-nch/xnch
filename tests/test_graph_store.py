"""Tests for the Kuzu-backed GraphStore (real embedded graph)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from xnch.memory.graph_store import GraphStore


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    g = GraphStore(tmp_path / "graph")
    g.connect()
    yield g
    g.close()


def test_upsert_entity(store):
    store.upsert_entity(id="svc-1", name="api-gateway", type_="service")
    ent = store.get_entity_by_name("api-gateway")
    assert ent is not None
    assert ent["metadata"]["entity_id"] == "svc-1"
    assert ent["metadata"]["type"] == "service"


def test_upsert_entity_update(store):
    store.upsert_entity(id="svc-1", name="api-gateway", type_="service")
    store.upsert_entity(id="svc-1", name="api-gateway-v2", type_="service")
    ent = store.get_entity_by_name("api-gateway-v2")
    assert ent is not None
    assert ent["metadata"]["entity_id"] == "svc-1"
    assert store.get_entity_by_name("api-gateway") is None


def test_get_entity_by_name(store):
    store.upsert_entity(id="svc-1", name="api-gateway", type_="service")
    entity = store.get_entity_by_name("api-gateway")
    assert entity is not None
    assert entity["document"] == "api-gateway"
    assert entity["metadata"]["entity_id"] == "svc-1"


def test_get_entity_by_name_case_insensitive(store):
    store.upsert_entity(id="svc-1", name="PaymentsService", type_="service")
    assert store.get_entity_by_name("paymentsservice") is not None


def test_get_entity_by_name_missing(store):
    assert store.get_entity_by_name("does-not-exist") is None


@pytest.mark.asyncio
async def test_upsert_relation(store):
    store.upsert_entity(id="usr-1", name="alice", type_="user")
    store.upsert_entity(id="svc-1", name="api-gateway", type_="service")
    await store.upsert_relation(from_id="usr-1", to_id="svc-1", rel_type="accessed", confidence=0.9)
    connections = store.query_entity_connections("usr-1")
    assert len(connections) == 1
    assert connections[0]["connected_name"] == "api-gateway"
    assert connections[0]["rel_type"] == "accessed"
    assert connections[0]["confidence"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_upsert_relation_updates_confidence(store):
    store.upsert_entity(id="usr-1", name="alice", type_="user")
    store.upsert_entity(id="svc-1", name="api-gateway", type_="service")
    await store.upsert_relation(from_id="usr-1", to_id="svc-1", rel_type="accessed", confidence=0.9)
    await store.upsert_relation(from_id="usr-1", to_id="svc-1", rel_type="accessed", confidence=0.95)
    connections = store.query_entity_connections("usr-1")
    assert len(connections) == 1
    assert connections[0]["confidence"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_upsert_relation_syncs_relationship_store(store, tmp_path):
    rel_store = AsyncMock()
    g = GraphStore(tmp_path / "rel-sync-test", relationship_store=rel_store)
    g.connect()
    g.upsert_entity(id="usr-1", name="alice", type_="user")
    g.upsert_entity(id="svc-1", name="api-gateway", type_="service")
    try:
        await g.upsert_relation(from_id="usr-1", to_id="svc-1", rel_type="accessed", confidence=0.9)
    finally:
        g.close()
    rel_store.upsert_relationship.assert_awaited_once()
    _, kwargs = rel_store.upsert_relationship.call_args
    assert kwargs["entity_a"] == "usr-1"
    assert kwargs["entity_b"] == "svc-1"
    assert kwargs["strength"] == 0.9


def test_query_entity_connections_both_directions(store):
    store.upsert_entity(id="usr-1", name="alice", type_="user")
    store.upsert_entity(id="svc-1", name="api-gateway", type_="service")
    store.upsert_entity(id="svc-2", name="auth-gateway", type_="service")
    import asyncio

    async def _seed():
        await store.upsert_relation(from_id="usr-1", to_id="svc-1", rel_type="accessed", confidence=0.9)
        await store.upsert_relation(from_id="svc-2", to_id="usr-1", rel_type="monitored_by", confidence=0.6)

    asyncio.run(_seed())
    connections = store.query_entity_connections("usr-1")
    assert len(connections) == 2
    rel_types = {c["rel_type"] for c in connections}
    assert rel_types == {"accessed", "monitored_by"}


def test_query_entity_connections_empty(store):
    assert store.query_entity_connections("ghost") == []


def test_db_path_isolation(tmp_path: Path) -> None:
    ga = GraphStore(tmp_path / "graph_a")
    gb = GraphStore(tmp_path / "graph_b")
    ga.connect()
    gb.connect()
    ga.upsert_entity(id="svc-1", name="api-gateway", type_="service")
    assert gb.get_entity_by_name("api-gateway") is None
    ga.close()
    gb.close()


def test_no_connect_guards(tmp_path: Path) -> None:
    g = GraphStore(tmp_path / "graph")
    assert g.get_entity_by_name("x") is None
    assert g.query_entity_connections("x") == []
    assert g.fetch_entities(limit=5) == []
    g.upsert_entity(id="1", name="n", type_="t")
    g.close()


def test_fetch_entities_orders_by_recency(store) -> None:
    for i in range(5):
        store.upsert_entity(id=f"e{i}", name=f"entity{i}", type_="svc")
    entities = store.fetch_entities(limit=3)
    assert [e["document"] for e in entities] == ["entity4", "entity3", "entity2"]


def test_fetch_entities_deterministic_tiebreak(store) -> None:
    for eid in ("beta", "alpha", "gamma"):
        store.upsert_entity(id=eid, name=f"entity-{eid}", type_="svc")
    store._conn.execute("MATCH (e:entities) SET e.created_at = timestamp('2024-01-01T00:00:00')")
    entities = store.fetch_entities(limit=10)
    ids = [e["metadata"]["entity_id"] for e in entities]
    assert ids == ["alpha", "beta", "gamma"]
    assert store.fetch_entities(limit=10) == entities
