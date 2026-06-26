"""Tests for SensoryBuffer (Layer 0) with fakeredis."""

from __future__ import annotations

import pytest

from xnch.memory.sensory_buffer import SensoryBuffer
from xnch.memory.working_memory import WorkingMemory


@pytest.fixture
async def fake_redis():
    import fakeredis.aioredis
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield r


@pytest.fixture
async def buffer(fake_redis):
    sb = SensoryBuffer(redis_client=fake_redis)
    yield sb
    await sb.aclose()


@pytest.mark.asyncio
async def test_write_perception(buffer):
    key = await buffer.write_perception("voice", b"hello world", ttl=60)
    assert key.startswith("perception:voice:")
    assert await buffer._redis.get(key) is not None


@pytest.mark.asyncio
async def test_write_perception_str_data(buffer):
    key = await buffer.write_perception("keyboard", "hello from keyboard", ttl=60)
    assert key.startswith("perception:keyboard:")
    assert await buffer._redis.get(key) is not None


@pytest.mark.asyncio
async def test_write_perception_ttl(buffer):
    key = await buffer.write_perception("vision", "snapshot", ttl=1)
    assert await buffer._redis.ttl(key) <= 1


@pytest.mark.asyncio
async def test_read_recent(buffer, fake_redis):
    await buffer.write_perception("voice", "first", ttl=60)
    await buffer.write_perception("voice", "second", ttl=60)
    await buffer.write_perception("vision", "screen", ttl=60)

    voice = await buffer.read_recent("voice", limit=10)
    assert len(voice) == 2
    assert all(p["source"] == "voice" for p in voice)

    vision = await buffer.read_recent("vision", limit=10)
    assert len(vision) == 1
    assert vision[0]["data"] == "screen"


@pytest.mark.asyncio
async def test_read_recent_empty_source(buffer):
    result = await buffer.read_recent("voice", limit=10)
    assert result == []


@pytest.mark.asyncio
async def test_read_recent_limit(buffer):
    for i in range(5):
        await buffer.write_perception("voice", f"msg-{i}", ttl=60)
    result = await buffer.read_recent("voice", limit=3)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_flush_to_working_memory(buffer, fake_redis):
    key = await buffer.write_perception("voice", "hello there", ttl=60)
    wm = WorkingMemory(redis_client=fake_redis)
    await buffer.flush_to_working_memory(key, "greeting", working_memory=wm)
    assert await buffer._redis.get(key) is None
    turns = await wm.get_turns("default")
    assert len(turns) == 1
    assert turns[0]["role"] == "perception"
    assert "[voice] greeting" in turns[0]["content"]


@pytest.mark.asyncio
async def test_flush_to_working_memory_no_wm(buffer):
    key = await buffer.write_perception("file", "data", ttl=60)
    await buffer.flush_to_working_memory(key, "summary")
    assert await buffer._redis.get(key) is None


@pytest.mark.asyncio
async def test_flush_missing_key(buffer):
    await buffer.flush_to_working_memory("perception:voice:nonexistent", "nope")
