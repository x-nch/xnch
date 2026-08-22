"""Store injection into pipeline graph nodes.

Regression: assemble_context hardcoded every store to None, so live
invokes crashed with 'NoneType' object has no attribute 'get_turns'
while stubbed-graph route tests stayed green.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

import xnch.agents.pipeline_graph as pg


@pytest.fixture()
def fake_stores() -> dict[str, SimpleNamespace]:
    """Sentinel stores — identity must survive injection."""
    return {
        "working_memory": SimpleNamespace(tag="wm"),
        "pg_episodic": SimpleNamespace(tag="episodic"),
        "graph_store": SimpleNamespace(tag="graph"),
        "relationship_store": SimpleNamespace(tag="rel"),
        "sensory_buffer": SimpleNamespace(tag="sensory"),
    }


def _state() -> dict:
    from uuid import uuid4

    return {"session_id": str(uuid4()), "raw_input": "Deploy edge-proxy service"}


async def test_assemble_context_receives_injected_stores(
    fake_stores: dict[str, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Node forwards bound stores to nexi's assembler (identity preserved)."""
    captured: dict = {}

    async def fake_assemble(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            system_prompt="",
            recent_turns=[],
            relevant_episodes=[],
            entity_context=[],
            relationship_context=[],
            perception_snippets=[],
        )

    monkeypatch.setattr(
        "nexi.pipeline.context_assembler.assemble_context", fake_assemble
    )
    node = pg._make_context_node(fake_stores)
    await node(_state())

    for name, store in fake_stores.items():
        assert captured[name] is store, f"{name} not injected"


async def test_make_context_node_defaults_when_stores_missing() -> None:
    """Empty/missing stores bind explicit Nones (no silent NameError)."""
    node = pg._make_context_node({})
    assert node.keywords["working_memory"] is None
    assert node.keywords["sensory_buffer"] is None


def test_create_pipeline_wires_context_node_with_stores(
    fake_stores: dict[str, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_pipeline forwards the stores dict into the context node factory."""
    captured: dict = {}
    real_factory = pg._make_context_node

    def spy(stores):
        captured["stores"] = stores
        return real_factory(stores)

    monkeypatch.setattr(pg, "_make_context_node", spy)
    pg.create_pipeline(checkpointer=None, stores=fake_stores)
    assert captured["stores"] is fake_stores
