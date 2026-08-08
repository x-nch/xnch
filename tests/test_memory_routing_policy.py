"""Tests for memory routing policy loader."""

from pathlib import Path

from xnch.memory.routing_policy import load_memory_routing_policy, memory_target_for_tool


def test_load_memory_routing_policy(tmp_path: Path):
    path = tmp_path / "memory-routing.yaml"
    path.write_text(
        """
primary: xnch_episodic
curated: agentmemory
deprecate_store_note_for: [nexi, agent]
"""
    )
    policy = load_memory_routing_policy(path)
    assert policy.primary == "xnch_episodic"
    assert policy.curated == "agentmemory"
    assert policy.deprecate_store_note_for == frozenset({"nexi", "agent"})


def test_memory_target_for_tool_names():
    assert memory_target_for_tool("xnch_memory_surface") == "episodic"
    assert memory_target_for_tool("am_memory_lesson_save") == "agentmemory"
