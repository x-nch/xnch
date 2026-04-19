"""Step 6: /policy/check — dry-run policy evaluation per option."""
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/policy", tags=["policy"])


class PolicyCheckRequest(BaseModel):
    session_id: str
    system_state_version: str
    actor_role: str
    option_id: str
    action: dict[str, Any]


@router.get("/check")
@router.post("/check")
async def policy_check(body: PolicyCheckRequest, request: Request) -> dict[str, Any]:
    """Contract 1: evaluate action against active policy set."""
    app = request.app.state
    action = body.action

    result = app.policy_engine.evaluate(
        intent_class=action.get("intent_class", ""),
        action_type=action.get("type", ""),
        entity_class=action.get("entity_class", ""),
        actor_role=body.actor_role,
        actor_capabilities=action.get("actor_capabilities", []),
        action_spec=action.get("spec", {}),
        urgency=action.get("urgency", "NORMAL"),
        reversible=action.get("reversible", True),
    )

    response: dict[str, Any] = {
        "option_id": body.option_id,
        "session_id": body.session_id,
        "verdict": result.verdict,
        "policy_refs": result.policy_refs,
        "warnings": result.warnings,
        "modified_action_spec": result.modified_action_spec,
        "requires_actor": result.requires_actor,
    }

    app.event_log.emit(
        body.session_id, "xnch.policy", "DRY_RUN_EVALUATED",
        data={"option_id": body.option_id, "verdict": result.verdict},
    )
    return response
