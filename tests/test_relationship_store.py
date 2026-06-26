"""Tests for RelationshipStore with mocked asyncpg."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xnch.memory.relationship_store import RelationshipRecord, RelationshipStore


@pytest.fixture
async def store():
    s = RelationshipStore("postgresql://localhost:5432/xnch")
    conn = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    pool.acquire.return_value.__aexit__.return_value = None
    pool.close = AsyncMock()
    s._pool = pool
    yield s


@pytest.mark.asyncio
async def test_upsert_relationship(store):
    await store.upsert_relationship("alice", "bob", "knows", "met at conference", strength=0.9)
    async with store._pool.acquire() as conn:
        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert "INSERT INTO relationship_memory" in args[0]
        assert args[1] == "alice"
        assert args[2] == "bob"


@pytest.mark.asyncio
async def test_upsert_relationship_no_pool():
    s = RelationshipStore("postgresql://localhost:5432/xnch")
    await s.upsert_relationship("a", "b", "knows", "evidence")


@pytest.mark.asyncio
async def test_get_relationships(store):
    mock_rows = [
        {
            "entity_a_id": "alice",
            "entity_b_id": "bob",
            "relationship_type": "knows",
            "strength": 0.9,
            "reinforcement_count": 3,
        },
        {
            "entity_a_id": "alice",
            "entity_b_id": "charlie",
            "relationship_type": "works_with",
            "strength": 0.7,
            "reinforcement_count": 1,
        },
    ]
    async with store._pool.acquire() as conn:
        conn.fetch = AsyncMock(return_value=mock_rows)
    results = await store.get_relationships("alice")
    assert len(results) == 2
    assert isinstance(results[0], RelationshipRecord)
    assert results[0].entity_a_id == "alice"
    assert results[0].entity_b_id == "bob"
    assert results[0].relationship_type == "knows"
    assert results[0].strength == 0.9
    assert results[0].reinforcement_count == 3


@pytest.mark.asyncio
async def test_get_relationships_no_pool():
    s = RelationshipStore("postgresql://localhost:5432/xnch")
    results = await s.get_relationships("alice")
    assert results == []


@pytest.mark.asyncio
async def test_get_relationship_strength(store):
    async with store._pool.acquire() as conn:
        conn.fetchrow = AsyncMock(return_value={"strength": 0.85})
    strength = await store.get_relationship_strength("alice", "bob")
    assert strength == 0.85


@pytest.mark.asyncio
async def test_get_relationship_strength_none(store):
    async with store._pool.acquire() as conn:
        conn.fetchrow = AsyncMock(return_value=None)
    strength = await store.get_relationship_strength("alice", "bob")
    assert strength is None


@pytest.mark.asyncio
async def test_get_relationship_strength_no_pool():
    s = RelationshipStore("postgresql://localhost:5432/xnch")
    strength = await s.get_relationship_strength("alice", "bob")
    assert strength is None


@pytest.mark.asyncio
async def test_connect_creates_pool():
    with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_pool:
        mock_pool.return_value = AsyncMock()
        s = RelationshipStore("postgresql://localhost:5432/xnch")
        await s.connect()
        mock_pool.assert_called_once_with(
            "postgresql://localhost:5432/xnch",
            min_size=1,
            max_size=5,
        )
        await s.close()


@pytest.mark.asyncio
async def test_close(store):
    mock_pool = store._pool
    await store.close()
    mock_pool.close.assert_called_once()
    assert store._pool is None
