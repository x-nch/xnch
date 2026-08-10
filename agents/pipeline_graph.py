"""LangGraph decision pipeline for XNCH/Nexi.

Node functions are thin wrappers that delegate to existing pipeline modules.
Runtime dependencies (stores, adapters) are injected via PipelineDeps at compile time.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .decision_state import DecisionState
from .deps import PipelineDeps


def _session_from_state(state: DecisionState) -> Any:
    from nexi.models import Actor, ActorRole, SessionContext

    actor_data = state.get("actor") or {}
    return SessionContext(
        session_id=UUID(state["session_id"]),
        trace_id=UUID(state["trace_id"]),
        actor=Actor(
            id=actor_data.get("id", "unknown"),
            role=ActorRole(actor_data.get("role", "OPERATOR")),
            capability_set=actor_data.get("capability_set", []),
        ),
        system_state_version=state.get("system_state_version", ""),
        policy_version=state.get("policy_version", ""),
        idempotency_key=UUID(state.get("idempotency_key", str(uuid4()))),
        raw_input=state["raw_input"],
        priority=state.get("priority", "NORMAL"),
    )


def create_pipeline(deps: PipelineDeps | None = None, checkpointer=None):
    """Build and compile the LangGraph decision pipeline."""
    deps = deps or PipelineDeps()

    async def classify_intent(state: DecisionState) -> dict[str, Any]:
        """Node: classify user intent (maps from intent_interpreter.py)."""
        from nexi.pipeline.intent_interpreter import IntentInterpreter

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
            "events": [{"type": "intent_classified", "intent_class": str(intent.intent_class)}],
        }

    async def assemble_context(state: DecisionState) -> dict[str, Any]:
        """Node: assemble context (maps from context_assembler.py)."""
        from nexi.pipeline.context_assembler import assemble_context as _assemble

        ctx = await _assemble(
            session_id=state["session_id"],
            raw_input=state["raw_input"],
            working_memory=deps.working_memory,
            pg_episodic=deps.pg_episodic,
            graph_store=deps.graph_store,
            relationship_store=deps.relationship_store,
            sensory_buffer=deps.sensory_buffer,
            proactivity_engine=deps.proactivity_engine,
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
        from nexi.adapters.model_adapter import ModelAdapter
        from nexi.models import ContextManifest, Intent
        from nexi.pipeline.option_generator import generate_options as _generate

        adapter = ModelAdapter()
        session = _session_from_state(state)
        intent = Intent(**state["intent"])
        manifest = ContextManifest(**state["context"])

        options, path = await _generate(
            adapter=adapter,
            session=session,
            intent=intent,
            manifest=manifest,
        )
        return {
            "options": [o.model_dump(mode="json") for o in options],
            "events": [{"type": "options_generated", "count": len(options), "path": str(path)}],
        }

    async def filter_policy(state: DecisionState) -> dict[str, Any]:
        """Node: filter options through policy engine."""
        from nexi.adapters.xnch_client import XnchClient
        from nexi.models import PlanOption
        from nexi.pipeline.policy_filter import PolicyFilter

        session = _session_from_state(state)
        options = [PlanOption(**o) for o in state["options"]]
        intent = state.get("intent") or {}
        actor = state.get("actor") or {}

        if deps.policy_engine is not None:
            verdicts = []
            for opt in options:
                result = deps.policy_engine.evaluate(
                    intent_class=str(intent.get("intent_class", "")),
                    action_type=opt.action_type,
                    entity_class=str(intent.get("target_entity_class", "")),
                    actor_role=str(actor.get("role", "OPERATOR")),
                    actor_capabilities=actor.get("capability_set", []),
                    action_spec=opt.action_spec.model_dump()
                    if hasattr(opt.action_spec, "model_dump")
                    else opt.action_spec,
                    urgency=str(intent.get("urgency", "NORMAL")),
                    reversible=opt.reversible,
                )
                verdicts.append({
                    "option_id": str(opt.option_id),
                    "verdict": result.verdict,
                    "policy_refs": result.policy_refs,
                    "warnings": result.warnings,
                    "modified_action_spec": result.modified_action_spec,
                })
            return {"policy_verdicts": verdicts}

        xnch = XnchClient()
        filter_ = PolicyFilter(xnch)
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
        for v in state.get("policy_verdicts") or []:
            if v.get("verdict") == "BLOCK":
                return "end"
        return "evaluate"

    async def evaluate(state: DecisionState) -> dict[str, Any]:
        """Node: score and evaluate options (maps from evaluator.py)."""
        from nexi.models import (
            ContextManifest,
            EvaluatedOption,
            Intent,
            PlanOption,
            PolicyDryRunResponse,
        )
        from nexi.pipeline.evaluator import Evaluator

        evaluator = Evaluator()
        session = _session_from_state(state)
        intent = Intent(**state["intent"])
        manifest = ContextManifest(**state["context"])

        paired = []
        opt_map = {v.get("option_id"): v for v in state.get("policy_verdicts") or []}
        for o in state["options"]:
            opt = PlanOption(**o)
            v_data = opt_map.get(str(opt.option_id), {})
            verdict = PolicyDryRunResponse(
                option_id=opt.option_id,
                session_id=session.session_id,
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
            "evaluated": [e.model_dump(mode="json") for e in evaluated],
            "events": [{"type": "options_evaluated", "count": len(evaluated)}],
        }

    async def select(state: DecisionState) -> dict[str, Any]:
        """Node: select best option. May interrupt for human approval on EXECUTION."""
        from nexi.models import ContextManifest, EvaluatedOption, Intent, PlanOption
        from nexi.pipeline.selector import select_decision

        session = _session_from_state(state)
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
                    selected_option = o.model_dump(mode="json")
                    break

        intent_class = state["intent"].get("intent_class", "")
        if str(intent_class).endswith("EXECUTION") or intent_class == "EXECUTION":
            approved = interrupt({
                "action": "approve_execution",
                "selected": selected_option,
                "intent": state["intent"],
            })
            if not approved:
                return {
                    "selected": None,
                    "decision_id": str(record.decision_id),
                    "decision_record": record.model_dump(mode="json"),
                    "events": [{"type": "execution_rejected"}],
                }

        return {
            "selected": selected_option,
            "decision_id": str(record.decision_id),
            "decision_record": record.model_dump(mode="json"),
            "events": [{"type": "option_selected", "option_id": str(record.selected_option_id)}],
        }

    def route_after_select(state: DecisionState) -> str:
        """Conditional edge: route based on selection."""
        if state.get("selected") is None:
            return "end"
        return "compile_plan"

    async def compile_plan(state: DecisionState) -> dict[str, Any]:
        """Node: compile selected option into executable plan."""
        from nexi.models.options import PlanOption
        from nexi.pipeline.plan_compiler import compile_action_spec

        opt = PlanOption(**state["selected"])
        plan = compile_action_spec(opt)
        return {
            "compiled_plan": plan.model_dump(mode="json"),
            "events": [{"type": "plan_compiled"}],
        }

    async def dispatch(state: DecisionState) -> dict[str, Any]:
        """Node: mark plan ready for execution handoff."""
        return {
            "events": [{"type": "dispatched", "plan": state.get("compiled_plan")}],
        }

    graph = StateGraph(DecisionState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("assemble_context", assemble_context)
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
