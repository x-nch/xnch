"""Tests for PgEpisodicStore with mocked agentmemory."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from xnch.memory.pg_episodic_store import PgEpisodicStore


@pytest.fixture
async def store():
    s = PgEpisodicStore()
    await s.connect()
    yield s
    await s.close()


MOCK_CREATE = {
    "side_effect": lambda cat, text, id=None, embedding=None, metadata=None: id or "new-id"
}


@pytest.mark.asyncio
async def test_store_episode(store):
    with patch("xnch.memory.pg_episodic_store.create_memory") as mock_cm:
        mock_cm.return_value = "ep-001"
        eid = await store.store_episode(
            type_="decision",
            raw_text="deploy service foo",
            summary="deployment episode",
            importance=1.0,
        )
    assert eid is not None
    mock_cm.assert_called_once()
    _, kwargs = mock_cm.call_args
    assert kwargs["metadata"]["type"] == "decision"


@pytest.mark.asyncio
async def test_retrieve_similar(store):
    mock_results = [
        {
            "id": "m1",
            "document": "deploy service foo",
            "metadata": {
                "type": "decision",
                "raw_text": "deploy service foo",
                "summary": "",
                "importance": "1.0",
                "recall_count": "0",
                "last_recalled": "",
                "archived": "False",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decay_score": "1.0",
            },
            "embedding": None,
            "distance": 0.05,
        }
    ]
    with patch("xnch.memory.pg_episodic_store.get_memories", return_value=mock_results) as mock_gm:
        results = await store.retrieve_similar(top_k=5, min_score=0.5)
    assert len(results) == 1
    mock_gm.assert_called_once()


@pytest.mark.asyncio
async def test_bump_recall(store):
    with (
        patch("xnch.memory.pg_episodic_store.get_memory") as mock_gm,
        patch("xnch.memory.pg_episodic_store.update_memory") as mock_um,
    ):
        mock_gm.return_value = {
            "id": "m1",
            "document": "test",
            "metadata": {
                "type": "decision",
                "raw_text": "test",
                "importance": "1.0",
                "recall_count": "0",
                "last_recalled": "",
                "archived": "False",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
        await store.bump_recall("m1")
    mock_um.assert_called_once()
    args, kwargs = mock_um.call_args
    assert kwargs["metadata"]["recall_count"] == "1"


@pytest.mark.asyncio
async def test_list_recent(store):
    now = datetime.now(timezone.utc)
    mock_results = [
        {
            "id": "m1",
            "document": "ep1",
            "metadata": {
                "type": "decision",
                "raw_text": "ep1",
                "importance": "0.8",
                "recall_count": "1",
                "last_recalled": "",
                "decay_score": "0.8",
                "archived": "False",
                "timestamp": now.isoformat(),
            },
        }
    ]
    with patch("xnch.memory.pg_episodic_store.get_memories", return_value=mock_results):
        results = await store.list_recent(hours=24)
    assert len(results) == 1
    assert results[0]["type"] == "decision"


@pytest.mark.asyncio
async def test_retrieve_similar_empty(store):
    with patch("xnch.memory.pg_episodic_store.get_memories", return_value=[]):
        results = await store.retrieve_similar(top_k=5)
    assert results == []
