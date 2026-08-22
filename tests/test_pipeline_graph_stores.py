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


async def test_evaluate_passes_session_id_into_verdict(monkeypatch) -> None:
    """evaluate must rebuild PolicyDryRunResponse including session_id."""
    from uuid import uuid4

    captured: dict = {}

    class FakeEvaluator:
        def score(self, *, options, intent, manifest, session):
            captured["session"] = session
            return []

    monkeypatch.setattr("nexi.pipeline.evaluator.Evaluator", FakeEvaluator)
    sid = "11111111-1111-1111-1111-111111111111"
    await pg.evaluate({
        "session_id": sid,
        "trace_id": str(uuid4()),
        "raw_input": "x",
        "intent": {
            "session_id": sid,
            "intent_class": "EXECUTION", "action_type": "DEPLOY",
            "target_entity_id": "e", "target_entity_class": "service",
            "urgency": "NORMAL", "ambiguity_score": 0.0,
            "raw_input_hash": "sha256:x", "raw_input": "x",
        },
        "context": {"session_id": sid, "system_state_version": ""},
        "options": [{
            "option_id": "22222222-2222-2222-2222-222222222222",
            "action_type": "DEPLOY", "action_spec": {"type": "DEPLOY", "target": "edge-proxy", "params": {}},
            "stated_rationale": "deploy request", "reversible": True,
            "payload_hash": "sha256:y", "estimated_side_effects": [],
        }],
        "policy_verdicts": [{
            "option_id": "22222222-2222-2222-2222-222222222222",
            "session_id": sid,
            "verdict": "ALLOW", "policy_refs": [], "warnings": [],
            "modified_action_spec": None,
        }],
    })
    assert str(captured["session"].session_id) == sid
