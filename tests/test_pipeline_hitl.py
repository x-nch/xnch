"""HITL interrupt/resume for the LangGraph decision pipeline."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from xnch.agents.pipeline_runtime import PipelineRuntime


_SELECTED = {
    "option_id": str(uuid4()),
    "action_type": "apply",
    "action_spec": {"type": "noop", "target": "x", "params": {}},
    "reversible": True,
    "estimated_side_effects": [],
}


@pytest.fixture
def stubbed_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub pipeline nodes; keep interrupt semantics of select on EXECUTION."""
    import xnch.agents.pipeline_graph as pg

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
        return {
            "context": {
                "system_prompt": "",
                "recent_turns": [],
                "relevant_episodes": [],
                "entity_context": [],
                "relationship_context": [],
                "perception_snippets": [],
            }
        }

    async def generate_options(state: dict[str, Any]) -> dict[str, Any]:
        return {"options": [dict(_SELECTED)]}

    async def filter_policy(state: dict[str, Any]) -> dict[str, Any]:
        return {"policy_verdicts": [{"verdict": "ALLOW", "policy_refs": [], "warnings": [], "modified_action_spec": None}]}

    async def evaluate(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "evaluated": [
                {
                    "option_id": _SELECTED["option_id"],
                    "policy_verdict": "ALLOW",
                    "composite_score": 0.9,
                    "simulation_required": False,
                }
            ]
        }

    async def select(state: dict[str, Any]) -> dict[str, Any]:
        from xnch.agents.hitl import normalize_resume, should_interrupt_execution
        from xnch.config import settings as xnch_settings

        selected = state["options"][0] if state.get("options") else None
        if should_interrupt_execution(
            intent_class=str(state["intent"].get("intent_class", "")),
            evaluated=state.get("evaluated"),
            mode=xnch_settings.hitl_execution_mode,
            risk_threshold=xnch_settings.hitl_risk_threshold,
        ):
            decision = interrupt({
                "action": "approve_execution",
                "selected": selected,
                "intent": state["intent"],
                "decisions": ["approve", "reject"],
            })
            if not normalize_resume(decision):
                return {"selected": None, "events": [{"type": "execution_rejected"}]}
        return {
            "selected": selected,
            "events": [{"type": "option_selected", "option_id": selected["option_id"] if selected else None}],
        }

    async def compile_plan(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "compiled_plan": {"nodes": [{"action_type": "noop"}]},
            "events": [{"type": "plan_compiled"}],
        }

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


@pytest.mark.asyncio
async def test_execution_interrupt_then_approve(stubbed_pipeline: None) -> None:
    runtime = PipelineRuntime(checkpointer=MemorySaver())
    await runtime.start()
    try:
        out = await runtime.invoke(
            raw_input="apply change",
            session_id=str(uuid4()),
            thread_id="hitl-approve",
        )
        assert out["status"] == "interrupted"
        assert out["thread_id"] == "hitl-approve"
        assert out["interrupts"][0]["action"] == "approve_execution"

        resumed = await runtime.resume(thread_id="hitl-approve", decision="approve")
        assert resumed["status"] == "completed"
        assert resumed["approved"] is True
        assert resumed["decision"] == "approve"
        assert resumed["result"].get("compiled_plan") is not None
        events = [e.get("type") for e in resumed["result"].get("events", [])]
        assert "dispatched" in events
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_execution_interrupt_then_reject(stubbed_pipeline: None) -> None:
    runtime = PipelineRuntime(checkpointer=MemorySaver())
    await runtime.start()
    try:
        out = await runtime.invoke(
            raw_input="apply change",
            session_id=str(uuid4()),
            thread_id="hitl-reject",
        )
        assert out["status"] == "interrupted"

        resumed = await runtime.resume(thread_id="hitl-reject", decision="reject")
        assert resumed["status"] == "completed"
        assert resumed["decision"] == "reject"
        assert resumed["result"].get("selected") is None
        events = [e.get("type") for e in resumed["result"].get("events", [])]
        assert "execution_rejected" in events
        assert resumed["result"].get("compiled_plan") is None
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_hitl_mode_never_skips_interrupt(
    stubbed_pipeline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xnch.config import settings as xnch_settings

    monkeypatch.setattr(xnch_settings, "hitl_execution_mode", "never")
    runtime = PipelineRuntime(checkpointer=MemorySaver())
    await runtime.start()
    try:
        out = await runtime.invoke(
            raw_input="apply change",
            session_id=str(uuid4()),
            thread_id="hitl-never",
        )
        assert out["status"] == "completed"
        assert out["result"].get("compiled_plan") is not None
    finally:
        await runtime.stop()
