"""Invoke and resume the LangGraph decision pipeline from xnch."""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

from langgraph.types import Command

from nexi.models import (
    Actor,
    ActorRole,
    DecisionRecord,
    EvaluatedOption,
    Intent,
    PlanOption,
    SessionContext,
    VerdictResponse,
)
from nexi.pipeline import ClarificationRequired
from nexi.pipeline.dispatch import TokenExpired, dispatch_execution
from nexi.pipeline.plan_compiler import PlanCompilationError, compile_action_spec
from nexi.config import settings as nexi_settings

from ..auth.token import ExecutionTokenClaims
from .deps import PipelineDeps

logger = logging.getLogger(__name__)


class DecisionRunner:
    """Runs the compiled LangGraph pipeline and maps results to session responses."""

    def __init__(self, graph: Any, app_state: Any) -> None:
        self._graph = graph
        self._app = app_state

    def _thread_config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def run(
        self,
        *,
        raw_input: str,
        session_id: str,
        trace_id: str,
        actor: dict[str, Any],
        system_state_version: str,
        policy_version: str,
        idempotency_key: str,
        priority: str = "NORMAL",
    ) -> dict[str, Any]:
        initial_state = {
            "raw_input": raw_input,
            "session_id": session_id,
            "trace_id": trace_id,
            "actor": actor,
            "system_state_version": system_state_version,
            "policy_version": policy_version,
            "idempotency_key": idempotency_key,
            "priority": priority,
            "events": [],
        }
        config = self._thread_config(session_id)

        try:
            state = await self._graph.ainvoke(initial_state, config)
        except ClarificationRequired:
            return {
                "status": "CLARIFICATION_REQUIRED",
                "clarification_required": True,
                "session_id": session_id,
                "thread_id": session_id,
            }

        return await self._map_result(state, session_id, config)

    async def resume(self, thread_id: str, payload: Any) -> dict[str, Any]:
        config = self._thread_config(thread_id)
        state = await self._graph.ainvoke(Command(resume=payload), config)
        return await self._map_result(state, thread_id, config)

    async def get_thread_state(self, thread_id: str) -> dict[str, Any]:
        config = self._thread_config(thread_id)
        snapshot = await self._graph.aget_state(config)
        interrupts: list[Any] = []
        if snapshot.tasks:
            for task in snapshot.tasks:
                interrupts.extend(task.interrupts or [])
        return {
            "thread_id": thread_id,
            "values": snapshot.values,
            "next": snapshot.next,
            "interrupts": interrupts,
        }

    async def _map_result(
        self,
        state: dict[str, Any],
        thread_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot = await self._graph.aget_state(config)
        interrupts: list[Any] = []
        if snapshot.tasks:
            for task in snapshot.tasks:
                interrupts.extend(task.interrupts or [])

        if interrupts:
            return {
                "status": "AWAITING_APPROVAL",
                "session_id": thread_id,
                "thread_id": thread_id,
                "interrupt": interrupts[0].value if interrupts else None,
            }

        policy_verdicts = state.get("policy_verdicts") or []
        if policy_verdicts and all(v.get("verdict") == "BLOCK" for v in policy_verdicts):
            return {
                "status": "ESCALATED",
                "hold_id": str(uuid4()),
                "session_id": thread_id,
                "thread_id": thread_id,
            }

        if state.get("selected") is None:
            return {
                "status": "COMPLETED",
                "session_id": thread_id,
                "thread_id": thread_id,
                "intent": state.get("intent"),
                "events": state.get("events", []),
            }

        compiled_plan = state.get("compiled_plan")
        if not compiled_plan:
            return {
                "status": "COMPLETED",
                "session_id": thread_id,
                "thread_id": thread_id,
                "selected": state.get("selected"),
                "events": state.get("events", []),
            }

        return await self._finalize_execution(state, thread_id)

    async def _finalize_execution(self, state: dict[str, Any], session_id: str) -> dict[str, Any]:
        actor_data = state.get("actor") or {}
        session = SessionContext(
            session_id=UUID(session_id),
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

        selected = PlanOption(**state["selected"])
        intent = Intent(**state["intent"])
        if state.get("decision_record"):
            decision = DecisionRecord(**state["decision_record"])
        else:
            evaluated = [EvaluatedOption(**e) for e in state.get("evaluated", [])]
            from nexi.models import SelectionRationale

            decision = DecisionRecord(
                decision_id=UUID(state.get("decision_id") or str(uuid4())),
                session_id=session.session_id,
                intent_ref=UUID(str(uuid4())),
                context_manifest_ref=UUID(str(uuid4())),
                system_state_version=session.system_state_version,
                options_generated=len(state.get("options", [])),
                options_blocked=0,
                options_evaluated=evaluated,
                selected_option_id=selected.option_id,
                selection_rationale=SelectionRationale(
                    score_breakdown={},
                    weight_config_version="default",
                ),
                confidence=0.5,
                escalation_triggered=False,
            )

        evaluated = decision.options_evaluated

        try:
            compiled = compile_action_spec(selected)
        except PlanCompilationError as exc:
            return {
                "status": "ERROR",
                "error": str(exc),
                "session_id": session_id,
                "thread_id": session_id,
            }

        if not compiled.nodes:
            return {
                "status": "ERROR",
                "error": "compiled DAG has no nodes",
                "session_id": session_id,
                "thread_id": session_id,
            }

        node = compiled.nodes[0]
        validated_action_spec = {
            "type": node.action_type,
            "target": node.target,
            "params": node.params,
        }

        intent_class = intent.intent_class.value if hasattr(intent.intent_class, "value") else str(intent.intent_class)
        entity_class = intent.target_entity_class or ""
        opt_scores = {eo.option_id: eo.composite_score for eo in evaluated}
        outcome_score_predicted = opt_scores.get(selected.option_id, 0.5)

        verdict = await self._issue_verdict(
            session=session,
            decision=decision,
            validated_action_spec=validated_action_spec,
            payload_hash=selected.payload_hash,
            intent_class=intent_class,
            entity_class=entity_class,
            outcome_score_predicted=outcome_score_predicted,
        )

        if verdict.verdict == "BLOCK":
            return {
                "status": "ESCALATED",
                "hold_id": str(uuid4()),
                "session_id": session_id,
                "thread_id": session_id,
            }

        try:
            dispatch_payload = await dispatch_execution(
                session,
                decision,
                verdict,
                validated_action_spec,
                nexi_settings.execution_runner_url,
            )
        except TokenExpired:
            verdict = await self._issue_verdict(
                session=session,
                decision=decision,
                validated_action_spec=validated_action_spec,
                payload_hash=selected.payload_hash,
                intent_class=intent_class,
                entity_class=entity_class,
                outcome_score_predicted=outcome_score_predicted,
            )
            dispatch_payload = await dispatch_execution(
                session,
                decision,
                verdict,
                validated_action_spec,
                nexi_settings.execution_runner_url,
            )

        self._app.event_log.emit(
            str(session.trace_id),
            "xnch.decision",
            "EXECUTING",
            data={"execution_ref": str(dispatch_payload.execution_ref)},
        )

        return {
            "status": "EXECUTING",
            "session_id": session_id,
            "thread_id": session_id,
            "decision_id": str(decision.decision_id),
            "execution_ref": str(dispatch_payload.execution_ref),
            "audit_ref": str(verdict.audit_ref) if verdict.audit_ref else None,
        }

    async def _issue_verdict(
        self,
        *,
        session: SessionContext,
        decision: DecisionRecord,
        validated_action_spec: dict[str, Any],
        payload_hash: str,
        intent_class: str,
        entity_class: str,
        outcome_score_predicted: float,
    ) -> VerdictResponse:
        app = self._app
        current_version = await app.get_state_version()
        if session.system_state_version != current_version:
            session.system_state_version = current_version

        resolved = await app.governance.resolve_actor(session.actor.id)
        if not resolved:
            raise ValueError(f"Unknown actor: {session.actor.id}")

        result = app.policy_engine.evaluate(
            intent_class=intent_class,
            action_type=validated_action_spec.get("type", ""),
            entity_class=entity_class,
            actor_role=resolved.role,
            actor_capabilities=resolved.capability_set,
            action_spec=validated_action_spec.get("params", {}),
        )

        if result.verdict == "BLOCK":
            audit_ref = uuid4()
            app.ledger.write(
                decision_id=str(decision.decision_id),
                trace_id=str(session.trace_id),
                intent_hash=payload_hash,
                candidates_count=0,
                selected_option_id=None,
                scores={},
                audit_ref=str(audit_ref),
            )
            return VerdictResponse(
                request_id=decision.decision_id,
                verdict="BLOCK",
                verdict_reason=result.policy_refs[0] if result.policy_refs else "policy blocked",
                policy_refs=result.policy_refs,
                execution_token=None,
                token_ttl_ms=0,
                audit_ref=audit_ref,
            )

        policy_version = await app.get_policy_version()
        claims = ExecutionTokenClaims(
            session_id=str(session.session_id),
            decision_id=str(decision.decision_id),
            trace_id=str(session.trace_id),
            actor_id=resolved.id,
            actor_role=resolved.role,
            action_type=validated_action_spec.get("type", ""),
            entity_class=entity_class,
            policy_version=policy_version,
            system_state_version=current_version,
        )
        token, ttl_ms = app.token_signer.issue(claims)
        audit_ref = uuid4()

        app.ledger.write(
            decision_id=str(decision.decision_id),
            trace_id=str(session.trace_id),
            intent_hash=payload_hash,
            candidates_count=1,
            selected_option_id=str(decision.selected_option_id),
            scores={},
            audit_ref=str(audit_ref),
        )

        context_snapshot = {
            "session_id": str(session.session_id),
            "actor_id": resolved.id,
            "outcome_score_predicted": outcome_score_predicted,
        }
        await app.episodic.create_episode(
            decision_id=str(decision.decision_id),
            intent_class=intent_class,
            action_type=validated_action_spec.get("type", ""),
            entity_class=entity_class,
            actor_role=resolved.role,
            context_snapshot=context_snapshot,
            generation_path="MODEL",
        )
        await app.pg_episodic.store_decision_episode(
            decision_id=str(decision.decision_id),
            intent_class=intent_class,
            action_type=validated_action_spec.get("type", ""),
            entity_class=entity_class,
            actor_role=resolved.role,
            context_snapshot=context_snapshot,
            generation_path="MODEL",
        )

        return VerdictResponse(
            request_id=decision.decision_id,
            verdict=result.verdict,
            verdict_reason=result.policy_refs[0] if result.policy_refs else "allowed",
            policy_refs=result.policy_refs,
            modified_action=result.modified_action_spec,
            execution_token=token,
            token_ttl_ms=ttl_ms,
            audit_ref=audit_ref,
        )


def build_pipeline_deps(app_state: Any) -> PipelineDeps:
    """Construct pipeline dependencies from FastAPI app state."""
    return PipelineDeps(
        working_memory=app_state.working_memory,
        pg_episodic=app_state.pg_episodic,
        graph_store=app_state.graph_store,
        relationship_store=app_state.relationship_store,
        sensory_buffer=app_state.sensory_buffer,
        policy_engine=app_state.policy_engine,
    )
