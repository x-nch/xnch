"""LangGraph decision pipeline for XNCH/Nexi.

Node functions are thin wrappers that delegate to existing pipeline modules.
Runtime dependencies (stores, adapters) are injected via node kwargs at compile time.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..config import settings
from .decision_state import DecisionState
from .hitl import normalize_resume, should_interrupt_execution


async def classify_intent(state: DecisionState) -> dict[str, Any]:
    """Node: classify user intent (maps from intent_interpreter.py)."""
    from nexi.pipeline.intent_interpreter import IntentInterpreter
    from uuid import UUID

    interpreter = IntentInterpreter()
    intent = await interpreter.interpret(
        raw_input=state["raw_input"],
        session_id=UUID(state["session_id"]),
        trace_id=state["trace_id"],
    )
    return {
        "intent": {
            "intent_class": intent.intent_class,
            "action_type": intent.action_type,
            "target_entity_id": intent.target_entity_id,
            "target_entity_class": intent.target_entity_class,
            "urgency": intent.urgency,
            "ambiguity_score": intent.ambiguity_score,
            "raw_input": intent.raw_input,
        },
        "events": [{"type": "intent_classified", "intent_class": intent.intent_class}],
    }


async def assemble_context(
    state: DecisionState,
    *,
    working_memory: Any | None = None,
    pg_episodic: Any | None = None,
    graph_store: Any | None = None,
    relationship_store: Any | None = None,
    sensory_buffer: Any | None = None,
) -> dict[str, Any]:
    """Node: assemble context (maps from context_assembler.py).

    Stores are bound at compile time by _make_context_node(stores).
    """
    from nexi.pipeline.context_assembler import assemble_context as _assemble

    ctx = await _assemble(
        session_id=state["session_id"],
        raw_input=state["raw_input"],
        working_memory=working_memory,
        pg_episodic=pg_episodic,
        graph_store=graph_store,
        relationship_store=relationship_store,
        sensory_buffer=sensory_buffer,
    )
    return {
        "context": {
            "system_prompt": ctx.system_prompt,
            "recent_turns": ctx.recent_turns,
            "relevant_episodes": ctx.relevant_episodes,
            "entity_context": ctx.entity_context,
            "relationship_context": ctx.relationship_context,
            "perception_snippets": ctx.perception_snippets,
        },
    }


async def generate_options(state: DecisionState) -> dict[str, Any]:
    """Node: generate plan options (maps from option_generator.py)."""
    from nexi.pipeline.option_generator import generate_options as _generate
    from nexi.adapters.model_adapter import ModelAdapter
    from nexi.models import SessionContext, Intent, ContextManifest

    adapter = ModelAdapter()
    session = SessionContext(trace_id=state["trace_id"])
    intent = Intent(**state["intent"])
    manifest = ContextManifest(**state["context"])

    options, path = await _generate(
        adapter=adapter,
        session=session,
        intent=intent,
        manifest=manifest,
    )
    return {
        "options": [o.model_dump() for o in options],
        "events": [{"type": "options_generated", "count": len(options), "path": str(path)}],
    }


async def filter_policy(state: DecisionState) -> dict[str, Any]:
    """Node: filter options through policy engine (maps from policy_filter.py).

    PolicyFilter.filter() is async and requires SessionContext + PlanOption list.
    """
    from nexi.pipeline.policy_filter import PolicyFilter
    from nexi.adapters.xnch_client import XnchClient
    from nexi.models import SessionContext, PlanOption

    xnch = XnchClient()
    filter_ = PolicyFilter(xnch)
    session = SessionContext(trace_id=state["trace_id"])
    options = [PlanOption(**o) for o in state["options"]]

    surviving = await filter_.filter(session=session, options=options)

    verdicts = []
    for opt, resp in surviving:
        verdicts.append({
            "option_id": str(resp.option_id),
            "verdict": resp.verdict,
            "policy_refs": resp.policy_refs,
            "warnings": resp.warnings,
            "modified_action_spec": resp.modified_action_spec,
        })
    return {"policy_verdicts": verdicts}


def route_after_policy(state: DecisionState) -> str:
    """Conditional edge: route based on policy verdicts."""
    for v in state["policy_verdicts"]:
        if v.get("verdict") == "BLOCK":
            return "end"
    return "evaluate"


async def evaluate(state: DecisionState) -> dict[str, Any]:
    """Node: score and evaluate options (maps from evaluator.py)."""
    from nexi.pipeline.evaluator import Evaluator
    from nexi.models import SessionContext, Intent, ContextManifest, PlanOption, PolicyDryRunResponse

    evaluator = Evaluator()
    session = SessionContext(trace_id=state["trace_id"])
    intent = Intent(**state["intent"])
    manifest = ContextManifest(**state["context"])

    paired = []
    opt_map = {v.get("option_id"): v for v in state["policy_verdicts"]}
    for o in state["options"]:
        opt = PlanOption(**o)
        v_data = opt_map.get(str(opt.option_id), {})
        verdict = PolicyDryRunResponse(
            option_id=opt.option_id,
            verdict=v_data.get("verdict", "ALLOW"),
            policy_refs=v_data.get("policy_refs", []),
            warnings=v_data.get("warnings", []),
            modified_action_spec=v_data.get("modified_action_spec"),
        )
        paired.append((opt, verdict))

    evaluated = evaluator.score(
        options=paired,
        intent=intent,
        manifest=manifest,
        session=session,
    )
    return {
        "evaluated": [e.model_dump() for e in evaluated],
        "events": [{"type": "options_evaluated", "count": len(evaluated)}],
    }


async def select(state: DecisionState) -> dict[str, Any]:
    """Node: select best option. May interrupt for human approval on EXECUTION."""
    from nexi.pipeline.selector import select_decision
    from nexi.models import SessionContext, Intent, ContextManifest, PlanOption, EvaluatedOption

    session = SessionContext(trace_id=state["trace_id"])
    intent = Intent(**state["intent"])
    manifest = ContextManifest(**state["context"])
    options = [PlanOption(**o) for o in state["options"]]
    evaluated = [EvaluatedOption(**e) for e in state["evaluated"]]

    record = select_decision(
        session=session,
        intent=intent,
        manifest=manifest,
        options=options,
        evaluated=evaluated,
        n_generated=len(options),
        n_blocked=0,
        generation_path="MODEL",
    )

    selected_option = None
    if record.selected_option_id:
        for o in options:
            if str(o.option_id) == str(record.selected_option_id):
                selected_option = o.model_dump()
                break

    if state["intent"].get("intent_class") == "EXECUTION" and should_interrupt_execution(
        intent_class="EXECUTION",
        evaluated=state.get("evaluated"),
        mode=settings.hitl_execution_mode,
        risk_threshold=settings.hitl_risk_threshold,
    ):
        approved = normalize_resume(interrupt({
            "action": "approve_execution",
            "selected": selected_option,
            "intent": state["intent"],
        }))
        if not approved:
            return {"selected": None, "events": [{"type": "execution_rejected"}]}

    return {
        "selected": selected_option,
        "events": [{"type": "option_selected", "option_id": record.selected_option_id}],
    }


def route_after_select(state: DecisionState) -> str:
    """Conditional edge: route based on selection."""
    if state.get("selected") is None:
        return "end"
    return "compile_plan"


async def compile_plan(state: DecisionState) -> dict[str, Any]:
    """Node: compile selected option into executable plan."""
    from nexi.pipeline.plan_compiler import compile_action_spec
    from nexi.models.options import PlanOption

    opt = PlanOption(**state["selected"])
    plan = compile_action_spec(opt)
    return {
        "compiled_plan": plan.model_dump(),
        "events": [{"type": "plan_compiled"}],
    }


async def dispatch(state: DecisionState) -> dict[str, Any]:
    """Node: dispatch plan for execution."""
    return {
        "events": [{"type": "dispatched", "plan": state.get("compiled_plan")}],
    }


def _make_context_node(stores: dict[str, Any] | None) -> Any:
    """Bind real stores into assemble_context at graph compile time."""
    from functools import partial

    st = stores or {}
    return partial(
        assemble_context,
        working_memory=st.get("working_memory"),
        pg_episodic=st.get("pg_episodic"),
        graph_store=st.get("graph_store"),
        relationship_store=st.get("relationship_store"),
        sensory_buffer=st.get("sensory_buffer"),
    )


def create_pipeline(checkpointer=None, stores: dict[str, Any] | None = None):
    """Build and compile the LangGraph decision pipeline."""
    graph = StateGraph(DecisionState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("assemble_context", _make_context_node(stores))
    graph.add_node("generate_options", generate_options)
    graph.add_node("filter_policy", filter_policy)
    graph.add_node("evaluate", evaluate)
    graph.add_node("select", select)
    graph.add_node("compile_plan", compile_plan)
    graph.add_node("dispatch", dispatch)

    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "assemble_context")
    graph.add_edge("assemble_context", "generate_options")
    graph.add_edge("generate_options", "filter_policy")
    graph.add_conditional_edges(
        "filter_policy",
        route_after_policy,
        {"evaluate": "evaluate", "end": END},
    )
    graph.add_edge("evaluate", "select")
    graph.add_conditional_edges(
        "select",
        route_after_select,
        {"compile_plan": "compile_plan", "end": END},
    )
    graph.add_edge("compile_plan", "dispatch")
    graph.add_edge("dispatch", END)

    return graph.compile(checkpointer=checkpointer)
