"""Goal-step auto-dispatch — turns due goal steps into gated agent runs.

Flow (spec: docs/superpowers/specs/2026-08-23-agent-dispatch-design.md, v1 addendum):
cron claims a due goal, builds the Day-N prompt from simulation_plan, and files
a goal_step APPROVAL — the HITL gate. Approving spawns an agent_run linked to
the approval; its outcome posts the goal step-outcome. Nothing dispatches
without that human decision.

Enabled via XNCH_GOAL_DISPATCH_ENABLED; scoped by XNCH_GOAL_DISPATCH_GOAL_ID.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Plan entries declaring these kinds are always filed as risk_class='elevated'
# (2026-08-24 audit F6 addendum) — keyword allowlists cannot down-grade an
# explicitly external/consequential action.
ELEVATED_KINDS = frozenset(
    {
        "send_email", "submit_application", "purchase", "publish",
        "exec", "external_action", "delete",
    }
)


def _plan_of(goal: dict[str, Any]) -> list[Any]:
    """simulation_plan arrives as list (route-serialized) or JSON string (store row)."""
    plan = goal.get("simulation_plan") or []
    if isinstance(plan, str):
        import ast
        import json

        try:
            plan = json.loads(plan)
        except json.JSONDecodeError:
            try:
                plan = ast.literal_eval(plan)  # legacy python-repr rows
            except (ValueError, SyntaxError):
                return []
    return plan if isinstance(plan, list) else []


def build_step_prompt(goal: dict[str, Any], step_index: int) -> str | None:
    """Compose the opencode prompt for plan entry at step_index. None if exhausted."""
    plan = _plan_of(goal)
    if step_index >= len(plan):
        return None
    entry = plan[step_index]
    action = entry.get("action", "")
    output = entry.get("output", "deliverable.md")
    day = entry.get("day", f"step-{step_index + 1}")
    objective = (goal.get("objective") or "").strip()
    return (
        f"You are working one task of an active goal.\n"
        f"GOAL: {objective}\n"
        f"TASK ({day}): {action}\n"
        f"DELIVERABLE: produce the file '{output}' in the current working"
        f" directory with substantive, real content (research/analysis as"
        f" required). Use available tools and web search as needed."
        f" Do not ask questions — make reasonable assumptions and deliver."
    )


async def run_due_dispatch(
    *,
    goal_store,
    workflow_store,
    agent_run_store,
    goal_id: str,
    allowed_action_keywords: list[str] | None = None,
) -> dict[str, Any]:
    """One cron tick: maybe file a goal_step approval for the next plan step."""
    goal = await goal_store.get_goal(goal_id)
    if goal is None:
        return {"skipped": f"goal {goal_id} not found"}
    if goal.get("status") not in ("PENDING", "ACTIVE"):
        return {"skipped": f"goal status {goal.get('status')}"}
    done, max_steps = int(goal.get("steps_completed", 0)), int(goal["max_steps"])
    if done >= max_steps:
        return {"skipped": "goal steps exhausted"}

    pending = await workflow_store.pending_goal_approval(goal_id)
    if pending is not None:
        return {"skipped": "approval already pending"}

    latest = await workflow_store.latest_goal_approval(goal_id)
    if latest is not None and latest["status"] == "APPROVED":
        active = await agent_run_store.active_run_for_approval(latest["id"])
        if active is not None:
            return {"skipped": "previous step still in flight"}

    prompt = build_step_prompt(goal, done)
    if prompt is None:
        return {"skipped": "plan shorter than steps_completed"}

    entry_plan = _plan_of(goal)[done]
    action = str(entry_plan.get("action", ""))
    risk_class = "low"
    if allowed_action_keywords:
        lowered = action.lower()
        matched = any(
            k.strip().lower() in lowered
            for k in allowed_action_keywords
            if k.strip()
        )
        if not matched:
            risk_class = "elevated"
    # Explicit plan-entry signals force elevation regardless of keyword match
    # (2026-08-24 audit F6 addendum): a step that declares itself an external
    # action — or carries risk="elevated" — is never filed as low-risk.
    entry_kind = str(entry_plan.get("kind") or "").strip().lower()
    entry_risk = str(entry_plan.get("risk") or "").strip().lower()
    if entry_risk == "elevated" or entry_kind in ELEVATED_KINDS:
        risk_class = "elevated"
    approval = await workflow_store.create_goal_approval(
        goal_id=goal_id,
        payload={
            "goal_id": goal_id,
            "step_index": done,
            "day_label": entry_plan.get("day", f"step-{done + 1}"),
            "action": action,
            "output_file": entry_plan.get("output", ""),
            "prompt": prompt,
            "summary": f"[goal] {action[:120]}",
        },
        risk_class=risk_class,
    )
    logger.info("goal_dispatch: filed approval %s for goal %s step %d (risk=%s)",
                approval["id"], goal_id, done, risk_class)
    return {"approval_id": approval["id"], "step_index": done, "risk_class": risk_class}


async def retry_unspawned_approvals(
    *, workflow_store, agent_run_store
) -> dict[str, Any]:
    """Sweep: APPROVED goal_step approvals with no linked agent_run -> spawn.

    Covers the window where the decide-side hook failed after the decision was
    recorded. Idempotent: runs with any linked agent_run are skipped.
    """
    approved = await workflow_store.recent_goal_approvals(status="APPROVED", limit=20)
    spawned = 0
    for approval in approved:
        if await agent_run_store.get_run_by_approval(approval["id"]) is not None:
            continue
        await spawn_agent_run_for_approval(
            agent_run_store=agent_run_store,
            approval=approval,
            workflow_store=workflow_store,
        )
        spawned += 1
        logger.info("goal_dispatch: swept unspawned approval %s", approval["id"])
    return {"spawned": spawned}


async def spawn_agent_run_for_approval(
    *,
    agent_run_store,
    approval: dict[str, Any],
    workflow_store=None,
) -> dict[str, Any]:
    """Approve-side hook: turn an approved goal_step approval into an agent_run."""
    payload = approval.get("payload")
    if payload is None:
        payload = approval.get("payload_json")
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    run = await agent_run_store.create_run(
        prompt=payload["prompt"],
        workspace=None,
        approval_id=approval["id"],
    )
    if workflow_store is not None:
        try:
            await workflow_store.record_event(
                approval.get("producer_id", ""),
                "SPAWNED",
                actor="goal_dispatch",
                snapshot={
                    "run_id": run["id"],
                    "approval_id": approval["id"],
                    "step_index": payload.get("step_index"),
                },
            )
        except Exception:  # noqa: BLE001 — audit must not break dispatch
            logger.exception("failed to record SPAWNED event for %s", approval["id"])
    return run


async def apply_outcome_backpressure(
    *, agent_run_store, workflow_store, goal_store, run_row: dict[str, Any]
) -> None:
    """Terminal agent_run with a linked goal_step approval -> goal step-outcome."""
    if not run_row or not run_row.get("approval_id"):
        return
    approval = await workflow_store.get_approval(run_row["approval_id"])
    if approval is None:
        return
    try:
        await workflow_store.record_event(
            approval["producer_id"], run_row["status"],
            actor=run_row.get("runner_id") or "agent-runner",
            snapshot={
                "run_id": run_row["id"],
                "exit_code": run_row.get("exit_code"),
                "output_path": run_row.get("output_path"),
                "error": run_row.get("error"),
            },
        )
    except Exception:  # noqa: BLE001 — audit must not break back-pressure
        logger.exception("failed to record outcome event for run %s", run_row["id"])
    if approval["producer_type"] != "goal_step":
        return
    outcome = "SUCCESS" if run_row["status"] == "DONE" else "FAILURE"
    await goal_store.complete_step(approval["producer_id"], outcome)
