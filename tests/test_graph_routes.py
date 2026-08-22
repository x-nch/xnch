"""HTTP tests for Kuzu graph explorer routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from xnch.main import app
from xnch.memory.graph_store import GraphStore


@pytest.fixture
def graph_store(tmp_path: Path) -> GraphStore:
    graph = GraphStore(tmp_path / "graph")
    graph.connect()
    graph.upsert_entity(id="e1", name="xnch", type_="project")
    graph.upsert_entity(id="e2", name="kuzu", type_="library")
    yield graph
    graph.close()


@pytest.fixture
async def client(graph_store: GraphStore):
    app.state.graph_store = graph_store
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_graph_stats(client: AsyncClient, graph_store: GraphStore) -> None:
    import asyncio

    await graph_store.upsert_relation("e1", "e2", "uses", 0.95)
    await asyncio.sleep(0)
    resp = await client.get("/memory/graph/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["entity_count"] == 2
    assert data["relation_count"] == 1
    assert data["types"]["project"] == 1


@pytest.mark.asyncio
async def test_graph_entities_paginated(client: AsyncClient) -> None:
    resp = await client.get("/memory/graph/entities", params={"search": "kuz"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["entities"][0]["name"] == "kuzu"


@pytest.mark.asyncio
async def test_graph_subgraph(client: AsyncClient, graph_store: GraphStore) -> None:
    await graph_store.upsert_relation("e1", "e2", "uses", 0.95)
    resp = await client.get("/memory/graph/subgraph", params={"entity_id": "e1", "depth": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["center_id"] == "e1"
    assert len(data["entities"]) == 2
    assert len(data["relations"]) == 1
