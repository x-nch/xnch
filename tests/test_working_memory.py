"""Tests for WorkingMemory (Layer 1) with fakeredis."""

from __future__ import annotations

import pytest

from xnch.memory.working_memory import WorkingMemory


@pytest.fixture
async def fake_redis():
    import fakeredis.aioredis
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield r


@pytest.fixture
async def wm(fake_redis):
    w = WorkingMemory(redis_client=fake_redis)
    yield w
    await w.aclose()


@pytest.mark.asyncio
async def test_set_get_context(wm):
    await wm.set_context("session-1", "user_name", "Alice", ttl=3600)
    val = await wm.get_context("session-1", "user_name")
    assert val == "Alice"


@pytest.mark.asyncio
async def test_get_context_missing(wm):
    val = await wm.get_context("session-1", "nonexistent")
    assert val is None


@pytest.mark.asyncio
async def test_get_context_complex_value(wm):
    data = {"count": 3, "tags": ["a", "b"]}
    await wm.set_context("session-1", "prefs", data, ttl=3600)
    val = await wm.get_context("session-1", "prefs")
    assert val == data


@pytest.mark.asyncio
async def test_set_context_ttl(wm):
    await wm.set_context("session-1", "temp", "value", ttl=10)
    hkey = "session:session-1:temp"
    ttl = await wm._redis.ttl(hkey)
    assert 0 < ttl <= 10


@pytest.mark.asyncio
async def test_get_full_session(wm):
    await wm.set_context("sess-A", "topic", "AI", ttl=3600)
    await wm.set_context("sess-A", "model", "Gemma", ttl=3600)
    await wm.append_turn("sess-A", "user", "hello")
    full = await wm.get_full_session("sess-A")
    assert full["topic"] == "AI"
    assert full["model"] == "Gemma"
    assert "turns" not in full


@pytest.mark.asyncio
async def test_clear_session(wm):
    await wm.set_context("sess-B", "key1", "val1", ttl=3600)
    await wm.set_context("sess-B", "key2", "val2", ttl=3600)
    await wm.clear_session("sess-B")
    full = await wm.get_full_session("sess-B")
    assert full == {}


@pytest.mark.asyncio
async def test_clear_session_noop(wm):
    await wm.clear_session("nonexistent")


@pytest.mark.asyncio
async def test_append_turn(wm):
    await wm.append_turn("sess-C", "user", "hi")
    await wm.append_turn("sess-C", "assistant", "hello there")
    turns = await wm.get_turns("sess-C", last_n=20)
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == "hi"
    assert turns[1]["role"] == "assistant"
    assert "timestamp" in turns[0]


@pytest.mark.asyncio
async def test_get_turns_last_n(wm):
    for i in range(10):
        await wm.append_turn("sess-D", "user", f"msg-{i}")
    turns = await wm.get_turns("sess-D", last_n=3)
    assert len(turns) == 3
    assert turns[-1]["content"] == "msg-9"


@pytest.mark.asyncio
async def test_get_turns_empty(wm):
    turns = await wm.get_turns("empty-session", last_n=20)
    assert turns == []
