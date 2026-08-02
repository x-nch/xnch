"""Tests for PgEpisodicStore (asyncpg + pgvector) with mocked pool."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xnch.memory.pg_episodic_store import PgEpisodicStore


@pytest.fixture
def conn():
    c = MagicMock()
    c.execute = AsyncMock()
    c.fetch = AsyncMock()
    c.fetchrow = AsyncMock()
    return c


@pytest.fixture
def store(conn):
    s = PgEpisodicStore("postgresql://localhost:5432/xnch")
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    pool.acquire.return_value.__aexit__.return_value = None
    pool.close = AsyncMock()
    s._pool = pool
    return s


def _row(**overrides):
    now = datetime.now(timezone.utc)
    base = {
        "id": "e1",
        "type": "decision",
        "raw_text": "deploy service foo",
        "summary": "",
        "importance": 1.0,
        "recall_count": 0,
        "last_recalled": None,
        "timestamp": now,
        "decay_score": 1.0,
        "archived": False,
        "similarity": 0.95,
    }
    base.update(overrides)
    return base


def _drow(**overrides):
    now = datetime.now(timezone.utc)
    base = {
        "episode_id": "ep-9",
        "decision_id": "dec-1",
        "intent_class": "EXECUTION",
        "action_type": "deploy",
        "entity_class": "payments",
        "actor_role": "operator",
        "outcome": "SUCCESS",
        "prediction_delta": 0.05,
        "context_snapshot": {"scores": {"execution": 0.9}},
        "scores_json": '{"execution":0.9}',
        "generation_path": "MODEL",
        "created_at": now,
        "completed_at": now,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_connect_applies_schema():
    pool = MagicMock()
    c = MagicMock()
    c.execute = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = c
    pool.acquire.return_value.__aexit__.return_value = None
    pool.close = AsyncMock()
    with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_pool:
        mock_pool.return_value = pool
        s = PgEpisodicStore("postgresql://localhost:5432/xnch")
        await s.connect()
        assert s._pool is pool
        c.execute.assert_awaited_once()
        assert "CREATE EXTENSION IF NOT EXISTS vector" in c.execute.await_args.args[0]
        await s.close()
        assert s._pool is None


@pytest.mark.asyncio
async def test_store_episode_computes_embedding(store, conn):
    with patch("xnch.memory.pg_episodic_store.embed_text", return_value=[0.1, 0.2, 0.3]) as mock_embed:
        eid = await store.store_episode(
            type_="decision",
            raw_text="deploy service foo",
            summary="deployment episode",
            importance=1.0,
        )
    assert eid
    mock_embed.assert_called_once_with("deploy service foo")
    args = conn.execute.call_args[0]
    assert "INSERT INTO episodes" in args[0]
    assert args[2] == "decision"
    assert args[5] == "[0.1,0.2,0.3]"


@pytest.mark.asyncio
async def test_store_episode_explicit_embedding(store, conn):
    with patch("xnch.memory.pg_episodic_store.embed_text") as mock_embed:
        eid = await store.store_episode(
            type_="episode",
            raw_text="no embed needed",
            embedding=[0.5, 0.5, 0.5],
        )
    assert eid
    mock_embed.assert_not_called()
    assert conn.execute.call_args[0][5] == "[0.5,0.5,0.5]"


@pytest.mark.asyncio
async def test_retrieve_similar_semantic(store, conn):
    conn.fetch = AsyncMock(return_value=[_row()])
    with patch("xnch.memory.pg_episodic_store.embed_text", return_value=[0.1, 0.2]) as mock_embed:
        results = await store.retrieve_similar(query_text="deploy foo", top_k=3)
    assert len(results) == 1
    assert results[0]["id"] == "e1"
    assert results[0]["similarity"] == pytest.approx(0.95)
    mock_embed.assert_called_once_with("deploy foo")
    sql = conn.fetch.call_args[0][0]
    assert "<=>" in sql
    assert conn.fetch.call_args[0][1] == "[0.1,0.2]"
    assert conn.fetch.call_args[0][2] == 3


@pytest.mark.asyncio
async def test_retrieve_similar_filters_min_score(store, conn):
    conn.fetch = AsyncMock(return_value=[_row(similarity=0.3), _row(id="e2", similarity=0.8)])
    results = await store.retrieve_similar(
        embedding=[0.1, 0.2], top_k=5, min_score=0.5
    )
    assert [r["id"] for r in results] == ["e2"]


@pytest.mark.asyncio
async def test_retrieve_similar_recent_fallback(store, conn):
    conn.fetch = AsyncMock(return_value=[_row()])
    results = await store.retrieve_similar(top_k=5)
    assert len(results) == 1
    sql = conn.fetch.call_args[0][0]
    assert "<=>" not in sql
    assert "ORDER BY timestamp DESC" in sql


@pytest.mark.asyncio
async def test_retrieve_similar_no_pool():
    s = PgEpisodicStore("postgresql://localhost:5432/xnch")
    assert await s.retrieve_similar(query_text="x") == []


@pytest.mark.asyncio
async def test_bump_recall(store, conn):
    await store.bump_recall("e1")
    args = conn.execute.call_args[0]
    assert "recall_count = recall_count + 1" in args[0]
    assert args[1] == "e1"


@pytest.mark.asyncio
async def test_list_recent(store, conn):
    conn.fetch = AsyncMock(return_value=[_row()])
    results = await store.list_recent(hours=24)
    assert len(results) == 1
    assert results[0]["type"] == "decision"
    sql = conn.fetch.call_args[0][0]
    assert "timestamp >= $1" in sql
    assert isinstance(conn.fetch.call_args[0][1], datetime)


@pytest.mark.asyncio
async def test_store_decision_episode(store, conn):
    eid = await store.store_decision_episode(
        "dec-1", "EXECUTION", "deploy", "payments", "operator",
        context_snapshot={"scores": {"execution": 0.9}},
        scores_json='{"execution":0.9}',
    )
    assert eid
    args = conn.execute.call_args[0]
    assert "INSERT INTO decision_episodes" in args[0]
    assert args[2] == "dec-1"
    assert args[3] == "EXECUTION"


@pytest.mark.asyncio
async def test_complete_decision_episode(store, conn):
    conn.fetchrow = AsyncMock(return_value={"episode_id": "ep-9"})
    eid = await store.complete_decision_episode("dec-1", "SUCCESS", 0.05)
    assert eid == "ep-9"
    args = conn.execute.call_args[0]
    assert "SET outcome = $1" in args[0]


@pytest.mark.asyncio
async def test_complete_decision_episode_missing(store, conn):
    conn.fetchrow = AsyncMock(return_value=None)
    assert await store.complete_decision_episode("dec-x", "SUCCESS") is None


@pytest.mark.asyncio
async def test_fetch_decision_episodes_with_scores(store, conn):
    conn.fetch = AsyncMock(return_value=[_drow()])
    since = datetime.now(timezone.utc)
    results = await store.fetch_decision_episodes_with_scores(since)
    assert len(results) == 1
    assert results[0]["episode_id"] == "ep-9"
    sql = conn.fetch.call_args[0][0]
    assert "outcome IS NOT NULL" in sql
    assert "scores_json IS NOT NULL" in sql


@pytest.mark.asyncio
async def test_fetch_decision_episodes_since(store, conn):
    conn.fetch = AsyncMock(return_value=[_drow()])
    since = datetime.now(timezone.utc)
    results = await store.fetch_decision_episodes_since(since, limit=100)
    assert len(results) == 1
    assert conn.fetch.call_args[0][2] == 100


@pytest.mark.asyncio
async def test_fetch_for_manifest(store, conn):
    conn.fetch = AsyncMock(return_value=[_drow()])
    results = await store.fetch_for_manifest("EXECUTION", "payments", "operator")
    assert len(results) == 1
    assert results[0]["decision_id"] == "dec-1"
    assert results[0]["outcome"] == "SUCCESS"


@pytest.mark.asyncio
async def test_upsert_pattern(store, conn):
    await store.upsert_pattern(
        "sha256:x", "EXECUTION", "deploy", "payments", "operator",
        0.8, 0.9, 12, 0.03, "run-1",
    )
    args = conn.execute.call_args[0]
    assert "INSERT INTO patterns" in args[0]
    assert "ON CONFLICT (context_signature)" in args[0]
    assert args[2] == "sha256:x"


@pytest.mark.asyncio
async def test_fetch_patterns_for_manifest(store, conn):
    conn.fetch = AsyncMock(
        return_value=[
            {
                "pattern_id": "p-1",
                "context_signature": "sha256:x",
                "intent_class": "EXECUTION",
                "action_type": "deploy",
                "entity_class": "payments",
                "actor_role": "operator",
                "success_rate": 0.8,
                "confidence": 0.9,
                "observation_count": 12,
                "avg_prediction_delta": 0.03,
            }
        ]
    )
    results = await store.fetch_patterns_for_manifest("EXECUTION", "payments", "operator")
    assert len(results) == 1
    assert results[0]["pattern_id"] == "p-1"
    assert results[0]["confidence"] == 0.9


@pytest.mark.asyncio
async def test_fetch_patterns_low_success(store, conn):
    conn.fetch = AsyncMock(return_value=[])
    assert await store.fetch_patterns_low_success() == []


@pytest.mark.asyncio
async def test_no_pool_guards():
    s = PgEpisodicStore("postgresql://localhost:5432/xnch")
    assert await s.list_recent() == []
    assert await s.fetch_patterns_for_manifest("A", "B", "C") == []


@pytest.mark.asyncio
async def test_fetch_episodes_for_decay(store, conn):
    conn.fetch = AsyncMock(return_value=[_row(type="episode")])
    results = await store.fetch_episodes_for_decay(limit=500)
    assert len(results) == 1
    assert conn.fetch.call_args[0][1] == 500
    assert "archived = FALSE" in conn.fetch.call_args[0][0]


@pytest.mark.asyncio
async def test_apply_decay(store, conn):
    await store.apply_decay("e1", 0.05, True)
    sql, *rest = conn.execute.call_args[0]
    assert "UPDATE episodes" in sql
    assert rest[0] == "e1"
    assert rest[1] == 0.05
    assert rest[2] is True


@pytest.mark.asyncio
async def test_has_episode_of_type(store, conn):
    conn.fetchval = AsyncMock(return_value=1)
    assert await store.has_episode_of_type("identity") is True
    conn.fetchval = AsyncMock(return_value=None)
    assert await store.has_episode_of_type("identity") is False


@pytest.mark.asyncio
async def test_write_prediction_update(store, conn):
    await store.write_prediction_update("ep-9", 0.02, True)
    sql, episode_id, delta, flag = conn.execute.call_args[0]
    assert "UPDATE decision_episodes" in sql
    assert episode_id == "ep-9"
    assert delta == 0.02
    assert flag is True
