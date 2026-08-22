"""HTTP surface for the HITL pipeline: /governance/pipeline invoke/resume/status."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

import xnch.agents.pipeline_graph as pg
from xnch.agents.pipeline_runtime import PipelineRuntime
from xnch.routes.pipeline import router as pipeline_router


def _stub_pipeline_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub graph nodes; keep real interrupt semantics of select on EXECUTION."""
    selected = {
        "option_id": str(uuid4()),
        "action_type": "apply",
        "action_spec": {"type": "noop", "target": "x", "params": {}},
        "reversible": True,
        "estimated_side_effects": [],
    }

    async def classify_intent(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "intent": {
                "intent_class": "EXECUTION",
                "action_type": "apply",
                "target_entity_id": "e1",
                "target_entity_class": "service",
                "urgency": "normal",
                "ambiguity_score": 0.0,
                "raw_input": state["raw_input"],
            },
            "events": [{"type": "intent_classified"}],
        }

    async def assemble_context(state: dict[str, Any]) -> dict[str, Any]:
        return {"context": {"system_prompt": "", "recent_turns": [], "relevant_episodes": [], "entity_context": [], "relationship_context": [], "perception_snippets": []}}

    async def generate_options(state: dict[str, Any]) -> dict[str, Any]:
        return {"options": [dict(selected)]}

    async def filter_policy(state: dict[str, Any]) -> dict[str, Any]:
        return {"policy_verdicts": [{"verdict": "ALLOW", "policy_refs": [], "warnings": [], "modified_action_spec": None}]}

    async def evaluate(state: dict[str, Any]) -> dict[str, Any]:
        return {"evaluated": [{**selected, "composite_score": 0.9}], "events": [{"type": "options_evaluated"}]}

    async def select(state: dict[str, Any]) -> dict[str, Any]:
        from langgraph.types import interrupt

        from xnch.agents.hitl import normalize_resume

        approved = normalize_resume(interrupt({"action": "approve_execution", "selected": state["options"][0], "intent": state["intent"]}))
        if not approved:
            return {"selected": None, "events": [{"type": "execution_rejected"}]}
        return {"selected": state["options"][0], "events": [{"type": "option_selected"}]}

    async def compile_plan(state: dict[str, Any]) -> dict[str, Any]:
        return {"compiled_plan": {"nodes": [{"action_type": "noop"}]}, "events": [{"type": "plan_compiled"}]}

    async def dispatch(state: dict[str, Any]) -> dict[str, Any]:
        return {"events": [{"type": "dispatched"}]}

    monkeypatch.setattr(pg, "classify_intent", classify_intent)
    monkeypatch.setattr(pg, "assemble_context", assemble_context)
    monkeypatch.setattr(pg, "generate_options", generate_options)
    monkeypatch.setattr(pg, "filter_policy", filter_policy)
    monkeypatch.setattr(pg, "evaluate", evaluate)
    monkeypatch.setattr(pg, "select", select)
    monkeypatch.setattr(pg, "compile_plan", compile_plan)
    monkeypatch.setattr(pg, "dispatch", dispatch)


@pytest.fixture
def disabled_app() -> TestClient:
    """Router mounted without a runtime — mirrors XNCH_LANGGRAPH_PIPELINE=false."""
    app = FastAPI()
    app.include_router(pipeline_router)
    return TestClient(app)


@pytest.fixture
async def hitl_client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[TestClient]:
    """Router mounted with a started MemorySaver-backed pipeline runtime."""
    _stub_pipeline_nodes(monkeypatch)
    runtime = PipelineRuntime(checkpointer=MemorySaver())
    await runtime.start()

    app = FastAPI()
    app.include_router(pipeline_router)
    app.state.pipeline_runtime = runtime
    with TestClient(app) as client:
        yield client
    await runtime.stop()


def test_invoke_interrupts_then_resume_approves(hitl_client: TestClient) -> None:
    r = hitl_client.post(
        "/governance/pipeline/invoke",
        json={"session_id": str(uuid4()), "raw_input": "apply change", "thread_id": "api-1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "interrupted"
    assert body["interrupts"][0]["action"] == "approve_execution"

    pending = hitl_client.get("/governance/pipeline/api-1")
    assert pending.status_code == 200
    assert pending.json()["next"], "expected a pending resume node"

    resume = hitl_client.post("/governance/pipeline/resume", json={"thread_id": "api-1", "decision": "approve"})
    assert resume.status_code == 200
    done = resume.json()
    assert done["status"] == "completed"
    assert done["approved"] is True
    events = [e["type"] for e in done["result"]["events"]]
    assert "dispatched" in events


def test_resume_reject_blocks_dispatch(hitl_client: TestClient) -> None:
    hitl_client.post(
        "/governance/pipeline/invoke",
        json={"session_id": str(uuid4()), "raw_input": "apply change", "thread_id": "api-2"},
    )
    resume = hitl_client.post("/governance/pipeline/resume", json={"thread_id": "api-2", "decision": "reject"})
    assert resume.status_code == 200
    done = resume.json()
    assert done["decision"] == "reject"
    events = [e["type"] for e in done["result"]["events"]]
    assert "execution_rejected" in events


def test_resume_unknown_thread_404(hitl_client: TestClient) -> None:
    r = hitl_client.post("/governance/pipeline/resume", json={"thread_id": "ghost", "decision": "approve"})
    assert r.status_code == 404


def test_disabled_runtime_returns_503(disabled_app: TestClient) -> None:
    r = disabled_app.get("/governance/pipeline/some-thread")
    assert r.status_code == 503
    assert "XNCH_LANGGRAPH_PIPELINE" in r.json()["detail"]

    r = disabled_app.post("/governance/pipeline/invoke", json={"session_id": str(uuid4()), "raw_input": "x"})
    assert r.status_code == 503


def test_resume_requires_decision_or_approved(hitl_client: TestClient) -> None:
    r = hitl_client.post("/governance/pipeline/resume", json={"thread_id": "api-x"})
    assert r.status_code == 422


def test_invoke_missing_fields_422(disabled_app: TestClient) -> None:
    r = disabled_app.post("/governance/pipeline/invoke", json={"raw_input": "no session"})
    assert r.status_code == 422
