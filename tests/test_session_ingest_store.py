"""Tests for session-ingest extensions to PgEpisodicStore (mocked pool)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xnch.memory.pg_episodic_store import PgEpisodicStore


@pytest.fixture
def conn():
    c = MagicMock()
    c.execute = AsyncMock()
    c.fetch = AsyncMock(return_value=[])
    c.fetchval = AsyncMock(return_value=None)
    c.fetchrow = AsyncMock(return_value=None)
    return c


@pytest.fixture
def store(conn):
    s = PgEpisodicStore("postgresql://localhost/xnch")
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    pool.acquire.return_value.__aexit__.return_value = None
    pool.close = AsyncMock()
    s._pool = pool
    return s


async def test_connect_applies_ledger_schema(monkeypatch, conn):
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    pool.acquire.return_value.__aexit__.return_value = None
    pool.close = AsyncMock()

    async def fake_create_pool(*args, **kwargs):
        return pool

    monkeypatch.setattr(
        "xnch.memory.pg_episodic_store.asyncpg.create_pool", fake_create_pool
    )
    s = PgEpisodicStore("postgresql://localhost/xnch")
    await s.connect()
    statements = [call.args[0] for call in conn.execute.await_args_list]
    assert any("session_ingest_ledger" in stmt for stmt in statements)


async def test_store_session_episode_sets_timestamp(store, conn):
    ended = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    with patch(
        "xnch.memory.pg_episodic_store.embed_text", return_value=[0.1, 0.2, 0.3]
    ):
        await store.store_session_episode(
            raw_text="digest", summary="sum", timestamp=ended, importance=1.5
        )
    sql = conn.execute.await_args.args[0]
    assert "INSERT INTO episodes" in sql and "timestamp" in sql
    assert conn.execute.await_args.args[3] == ended


async def test_ledger_mark_done_upserts(store, conn):
    await store.ledger_mark_done("ses_a", "ep-uuid", facts_count=3)
    sql = conn.execute.await_args.args[0]
    assert "ON CONFLICT" in sql
    assert conn.execute.await_args.args[1:] == ("ses_a", "ep-uuid", 3)


async def test_ledger_mark_failed_records_error(store, conn):
    await store.ledger_mark_failed("ses_b", "ornith unreachable")
    args = conn.execute.await_args.args
    assert "FAILED" in args[0]
    assert args[1:] == ("ses_b", "ornith unreachable")


async def test_ledger_completed_ids_from_fetch(store, conn):
    conn.fetch.return_value = [
        {"session_id": "ses_a", "status": "SUCCEEDED"},
        {"session_id": "ses_c", "status": "SUCCEEDED"},
    ]
    done = await store.ledger_completed_ids()
    assert done == {"ses_a", "ses_c"}


async def test_ledger_get_returns_row_or_none(store, conn):
    conn.fetchrow.return_value = {
        "session_id": "ses_a",
        "episode_id": "ep-1",
        "status": "SUCCEEDED",
    }
    entry = await store.ledger_get("ses_a")
    assert entry is not None and entry["episode_id"] == "ep-1"
    conn.fetchrow.return_value = None
    assert await store.ledger_get("missing") is None
