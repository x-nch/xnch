"""Step 10: /verdict — authoritative policy check, Decision Ledger write, token issuance."""
import json
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..auth.token import ExecutionTokenClaims
from ..observability.langfuse_client import trace_llm_call
from ..observability.metrics import HITL_GATE_BYPASS

router = APIRouter(tags=["verdict"])


class VerdictRequest(BaseModel):
    request_id: str
    actor: dict[str, Any]
    action: dict[str, Any]
    context: dict[str, Any]


@router.post("/verdict")
async def verdict(body: VerdictRequest, request: Request) -> dict[str, Any]:
    """Step 10: authoritative policy re-evaluation, ledger write, token issuance."""
    app = request.app.state
    ctx = body.context

    # 1. Verify system_state_version match
    current_version = await app.get_state_version()
    submitted_version = ctx.get("system_state_version", "")
    if submitted_version != current_version:
        raise HTTPException(
            status_code=409,
            detail=f"STALE_SESSION: submitted version {submitted_version!r} "
                   f"!= current {current_version!r}",
        )

    # 2. Re-evaluate against active policy (authoritative — not dry-run)
    action = body.action
    actor = body.actor

    resolved = await app.governance.resolve_actor(actor.get("id", ""))
    if not resolved:
        raise HTTPException(status_code=401, detail="Unknown actor")

    result = app.policy_engine.evaluate(
        intent_class=action.get("intent_class", ""),
        action_type=action.get("type", ""),
        entity_class=action.get("entity_class", ""),
        actor_role=resolved.role,
        actor_capabilities=resolved.capability_set,
        action_spec=action.get("payload", {}),
    )

    if result.verdict == "BLOCK":
        audit_ref = str(uuid4())
        app.ledger.write(
            decision_id=body.request_id,
            trace_id=ctx.get("session_id", ""),
            intent_hash=action.get("payload_hash", ""),
            candidates_count=0,
            selected_option_id=None,
            scores={},
            audit_ref=audit_ref,
        )
        app.event_log.emit(ctx.get("session_id", ""), "xnch.verdict", "VERDICT_BLOCK",
                           data={"policy_refs": result.policy_refs})
        await trace_llm_call(
            prompt=json.dumps({"action": action, "actor": actor, "context": ctx}),
            response=json.dumps({"verdict": "BLOCK", "reason": result.policy_refs}),
            model="policy-engine",
            latency_ms=0,
            tokens_used=0,
            trace_id=ctx.get("session_id", ""),
        )
        return {
            "request_id": body.request_id,
            "verdict": "BLOCK",
            "verdict_reason": result.policy_refs[0] if result.policy_refs else "policy blocked",
            "policy_refs": result.policy_refs,
            "modified_action": None,
            "execution_token": None,
            "token_ttl_ms": 0,
            "audit_ref": audit_ref,
        }

    # 3. Issue execution token
    policy_version = await app.get_policy_version()
    claims = ExecutionTokenClaims(
        session_id=ctx.get("session_id", ""),
        decision_id=body.request_id,
        trace_id=ctx.get("session_id", ""),
        actor_id=resolved.id,
        actor_role=resolved.role,
        action_type=action.get("type", ""),
        entity_class=action.get("entity_class", ""),
        policy_version=policy_version,
        system_state_version=current_version,
    )
    token, ttl_ms = app.token_signer.issue(claims)

    # 4. Write to Decision Ledger (synchronous — before response)
    audit_ref = str(uuid4())
    app.ledger.write(
        decision_id=body.request_id,
        trace_id=ctx.get("session_id", ""),
        intent_hash=action.get("payload_hash", ""),
        candidates_count=1,
        selected_option_id=body.request_id,
        scores={},
        audit_ref=audit_ref,
    )

    app.event_log.emit(ctx.get("session_id", ""), "xnch.verdict", "VERDICT_ALLOW",
                       data={"audit_ref": audit_ref, "policy_refs": result.policy_refs})

    # Security signal: an EXECUTION decision allowed with a goal_id in context
    # was dispatched by the goal loop and never passed the HITL interrupt gate.
    if ctx.get("goal_id") and action.get("intent_class", "") == "EXECUTION":
        HITL_GATE_BYPASS.labels(origin="goal_loop").inc()
        app.event_log.emit(
            ctx.get("session_id", ""), "xnch.hitl", "HITL_GATE_BYPASS",
            level="WARN",
            data={
                "goal_id": ctx.get("goal_id"),
                "request_id": body.request_id,
                "intent_class": action.get("intent_class", ""),
                "action_type": action.get("type", ""),
            },
        )

    await trace_llm_call(
        prompt=json.dumps({"action": action, "actor": actor, "context": ctx}),
        response=json.dumps({"verdict": result.verdict, "token": token[:16] + "..." if token else ""}),
        model="policy-engine",
        latency_ms=0,
        tokens_used=0,
        trace_id=ctx.get("session_id", ""),
    )

    context_snapshot = {
        "session_id": ctx.get("session_id", ""),
        "actor_id": resolved.id,
        "outcome_score_predicted": ctx.get("outcome_score_predicted", 0.5),
        "goal_id": ctx.get("goal_id", ""),
    }
    await app.episodic.create_episode(
        decision_id=body.request_id,
        intent_class=body.action.get("intent_class", ""),
        action_type=body.action.get("type", ""),
        entity_class=body.action.get("entity_class", ""),
        actor_role=resolved.role,
        context_snapshot=context_snapshot,
        generation_path="MODEL",
    )
    await app.pg_episodic.store_decision_episode(
        decision_id=body.request_id,
        intent_class=body.action.get("intent_class", ""),
        action_type=body.action.get("type", ""),
        entity_class=body.action.get("entity_class", ""),
        actor_role=resolved.role,
        context_snapshot=context_snapshot,
        generation_path="MODEL",
    )

    return {
        "request_id": body.request_id,
        "verdict": result.verdict,
        "verdict_reason": result.policy_refs[0] if result.policy_refs else "allowed",
        "policy_refs": result.policy_refs,
        "modified_action": result.modified_action_spec,
        "execution_token": token,
        "token_ttl_ms": ttl_ms,
        "audit_ref": audit_ref,
    }
