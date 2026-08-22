"""Store injection into pipeline graph nodes.

Regression: assemble_context hardcoded every store to None, so live
invokes crashed with 'NoneType' object has no attribute 'get_turns'
while stubbed-graph route tests stayed green.
"""
from __future__ import annotations

import json

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
    st = _state()
    out = await node(st)

    assert out["context"]["session_id"] == st["session_id"]
    assert "system_state_version" in out["context"]
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


def test_session_from_state_builds_valid_context() -> None:
    """All four pipeline nodes need a complete SessionContext from DecisionState."""
    from uuid import UUID, uuid4

    state = {
        "session_id": str(uuid4()),
        "trace_id": str(uuid4()),
        "raw_input": "Deploy edge-proxy service",
    }
    session = pg._session_from_state(state)
    assert session.session_id == UUID(state["session_id"])
    assert session.trace_id == UUID(state["trace_id"])
    assert session.raw_input == state["raw_input"]
    assert session.actor.role.value in {"ADMIN", "OPERATOR", "VIEWER", "AGENT"}
    assert session.idempotency_key is not None


async def test_classify_intent_emits_complete_payload(monkeypatch) -> None:
    """classify_intent must emit every required Intent field (incl. session_id,
    raw_input_hash) so downstream nodes can rebuild the model."""
    from uuid import uuid4

    from nexi.models import Intent

    async def fake_interpret(self, *, raw_input, session_id, trace_id):
        return Intent(
            session_id=session_id,
            intent_class="EXECUTION",
            action_type="DEPLOY",
            target_entity_id="edge-proxy",
            target_entity_class="service",
            urgency="NORMAL",
            ambiguity_score=0.0,
            raw_input_hash="sha256:abc",
            raw_input=raw_input,
        )

    monkeypatch.setattr(
        "nexi.pipeline.intent_interpreter.IntentInterpreter.interpret",
        fake_interpret,
    )
    sid = str(uuid4())
    out = await pg.classify_intent(
        {"session_id": sid, "raw_input": "Deploy now", "trace_id": str(uuid4())}
    )
    payload = out["intent"]
    assert payload["session_id"] == sid
    assert payload["raw_input_hash"].startswith("sha256:")
    json.dumps(payload)
