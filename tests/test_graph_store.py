"""Tests for GraphStore with mocked agentmemory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from xnch.memory.graph_store import GraphStore


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    g = GraphStore(tmp_path / "graph")
    g.connect()
    yield g
    g.close()


MOCK_ENTITY = {
    "id": "svc-1",
    "document": "api-gateway",
    "metadata": {"entity_id": "svc-1", "name": "api-gateway", "type": "service"},
    "embedding": None,
}


def test_upsert_entity(store):
    with patch("xnch.memory.graph_store.search_memory") as mock_sm:
        mock_sm.return_value = []
        with patch("xnch.memory.graph_store.create_memory") as mock_cm:
            store.upsert_entity(id="svc-1", name="api-gateway", type_="service")
    mock_cm.assert_called_once()
    args, kwargs = mock_cm.call_args
    assert args[0] == "entities"
    assert args[1] == "api-gateway"
    assert kwargs["metadata"]["type"] == "service"


def test_upsert_entity_update(store):
    with patch("xnch.memory.graph_store.search_memory") as mock_sm:
        mock_sm.return_value = [MOCK_ENTITY]
        with patch("xnch.memory.graph_store.update_memory") as mock_um:
            store.upsert_entity(id="svc-1", name="api-gateway-v2", type_="service")
    mock_um.assert_called_once()


def test_get_entity_by_name(store):
    with patch("xnch.memory.graph_store.search_memory") as mock_sm:
        mock_sm.return_value = [MOCK_ENTITY]
        entity = store.get_entity_by_name("api-gateway")
    assert entity is not None
    assert entity["document"] == "api-gateway"


def test_get_entity_by_name_missing(store):
    with patch("xnch.memory.graph_store.search_memory") as mock_sm:
        mock_sm.return_value = []
        entity = store.get_entity_by_name("does-not-exist")
    assert entity is None


@pytest.mark.asyncio
async def test_upsert_relation(store):
    with patch("xnch.memory.graph_store.search_memory") as mock_sm:
        mock_sm.return_value = []
        with patch("xnch.memory.graph_store.create_memory") as mock_cm:
            await store.upsert_relation(from_id="usr-1", to_id="svc-1", rel_type="accessed", confidence=0.9)
        mock_cm.assert_called_once()
        args, kwargs = mock_cm.call_args
        assert "usr-1" in args[1]
        assert kwargs["metadata"]["rel_type"] == "accessed"


def test_query_entity_connections(store):
    mock_rels = [
        {
            "id": "r1",
            "document": "usr-1 accessed svc-1",
            "metadata": {"from_id": "usr-1", "to_id": "svc-1", "rel_type": "accessed", "confidence": "0.9"},
        }
    ]
    mock_entities = [
        {
            "id": "svc-1",
            "document": "api-gateway",
            "metadata": {"entity_id": "svc-1", "name": "api-gateway", "type": "service"},
        }
    ]
    with (
        patch("xnch.memory.graph_store.get_memories") as mock_gm,
        patch.object(store, "_get_entity_direct", return_value=mock_entities[0]),
    ):
        mock_gm.return_value = mock_rels
        connections = store.query_entity_connections("usr-1")
    assert len(connections) == 1
    assert connections[0]["connected_name"] == "api-gateway"
    assert connections[0]["rel_type"] == "accessed"


def test_db_path_isolation(tmp_path: Path) -> None:
    path_a = tmp_path / "graph_a"
    path_b = tmp_path / "graph_b"
    ga = GraphStore(path_a)
    ga.connect()
    gb = GraphStore(path_b)
    gb.connect()
    assert ga._path != gb._path
    ga.close()
    gb.close()
