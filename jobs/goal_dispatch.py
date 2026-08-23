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

    prompt = build_step_prompt(goal, done)
    if prompt is None:
        return {"skipped": "plan shorter than steps_completed"}

    entry_plan = _plan_of(goal)[done]
    approval = await workflow_store.create_goal_approval(
        goal_id=goal_id,
        payload={
            "goal_id": goal_id,
            "step_index": done,
            "day_label": entry_plan.get("day", f"step-{done + 1}"),
            "action": entry_plan.get("action", ""),
            "output_file": entry_plan.get("output", ""),
            "prompt": prompt,
            "summary": f"[goal] {entry_plan.get('action', '')[:120]}",
        },
    )
    logger.info("goal_dispatch: filed approval %s for goal %s step %d",
                approval["id"], goal_id, done)
    return {"approval_id": approval["id"], "step_index": done}


async def spawn_agent_run_for_approval(
    *, agent_run_store, approval: dict[str, Any]
) -> dict[str, Any]:
    """Approve-side hook: turn an approved goal_step approval into an agent_run."""
    payload = approval.get("payload")
    if payload is None:
        payload = approval.get("payload_json")
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    return await agent_run_store.create_run(
        prompt=payload["prompt"],
        workspace=None,
        approval_id=approval["id"],
    )


async def apply_outcome_backpressure(
    *, agent_run_store, workflow_store, goal_store, run_row: dict[str, Any]
) -> None:
    """Terminal agent_run with a linked goal_step approval -> goal step-outcome."""
    if not run_row.get("approval_id"):
        return
    approval = await workflow_store.get_approval(run_row["approval_id"])
    if approval is None or approval["producer_type"] != "goal_step":
        return
    outcome = "SUCCESS" if run_row["status"] == "DONE" else "FAILURE"
    await goal_store.complete_step(approval["producer_id"], outcome)
