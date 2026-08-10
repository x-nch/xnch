"""DecisionState — LangGraph state for the XNCH/Nexi decision pipeline."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class Intent(TypedDict):
    intent_class: str  # QUERY | DECISION | EXECUTION | ESCALATION
    action_type: str
    target_entity_id: str
    target_entity_class: str
    urgency: str
    ambiguity_score: float
    raw_input: str


class AssembledContext(TypedDict):
    system_prompt: str
    recent_turns: list[dict]
    relevant_episodes: list[str]
    entity_context: list[dict]
    relationship_context: list[dict]
    perception_snippets: list[str]


class PolicyVerdict(TypedDict):
    verdict: str  # ALLOW | ALLOW_WITH_WARNINGS | MODIFY | DEFER | BLOCK
    policy_refs: list[str]
    warnings: list[str]
    modified_action_spec: dict[str, Any] | None


class EvaluatedOption(TypedDict):
    option_id: str
    policy_verdict: str
    composite_score: float
    simulation_required: bool


class PlanOption(TypedDict):
    option_id: str
    action_type: str
    action_spec: dict[str, Any]
    reversible: bool
    estimated_side_effects: list[str]


class DecisionState(TypedDict, total=False):
    raw_input: str
    session_id: str
    trace_id: str
    actor: dict[str, Any]
    system_state_version: str
    policy_version: str
    idempotency_key: str
    priority: str

    intent: Intent
    context: AssembledContext
    options: list[PlanOption]
    policy_verdicts: list[PolicyVerdict]
    evaluated: list[EvaluatedOption]
    selected: PlanOption | None
    decision_id: str | None
    decision_record: dict[str, Any] | None
    compiled_plan: dict[str, Any] | None

    events: Annotated[list[dict[str, Any]], operator.add]
