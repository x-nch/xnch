"""Scheduled-workflow jobs on xnch's existing AsyncIOScheduler (P3).

Restart catch-up: jobs are registered with ``misfire_grace_time`` ≥ 1h and
``replace_existing=True``; lifespan re-runs :func:`sync_all_workflow_jobs` on
every boot, so a cron that fired while the process was down still executes
within the grace window.

Stdlib + apscheduler only; store injected for testability.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

JOB_ID_PREFIX = "workflow:"
MISFIRE_GRACE_S = 3600  # restart catch-up window


def _trigger_of(wf_row: dict[str, Any]) -> dict[str, Any]:
    raw = wf_row.get("trigger_json")
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}
    return raw or {}


def job_kwargs_for_trigger(wf_row: dict[str, Any]) -> dict[str, Any] | None:
    """APScheduler add_job kwargs for a scheduled workflow, else None."""
    trigger = _trigger_of(wf_row)
    if trigger.get("kind") != "schedule":
        return None
    cron = trigger.get("cron")
    if not cron:
        return None
    try:
        cron_trigger = CronTrigger.from_crontab(cron)
    except ValueError as exc:
        logger.warning(
            "workflow %s has invalid cron %r — not scheduled: %s",
            wf_row.get("id"),
            cron,
            exc,
        )
        return None
    return {
        "id": f"{JOB_ID_PREFIX}{wf_row['id']}",
        "trigger": cron_trigger,
        "misfire_grace_time": MISFIRE_GRACE_S,
        "replace_existing": True,
    }


def make_fire_fn(store: Any, workflow_id: str) -> Callable[..., Any]:
    """Build the async callable a cron job invokes to start one run."""

    async def fire() -> None:
        try:
            run, created = await store.create_run(
                workflow_id=workflow_id,
                actor="scheduler",
                trigger={"kind": "schedule"},
            )
            if created:
                logger.info("scheduled workflow fired (workflow=%s run=%s)", workflow_id, run["id"])
        except Exception as exc:
            logger.error("scheduled workflow %s failed to fire: %s", workflow_id, exc)

    return fire


def sync_workflow_job(scheduler: Any, wf_row: dict[str, Any], store: Any | None = None) -> None:
    """(Re-)register or remove the cron job for one workflow definition."""
    wf_id = wf_row["id"]
    remove_workflow_job(scheduler, wf_id)
    kwargs = job_kwargs_for_trigger(wf_row)
    if kwargs is None:
        return
    if store is None:
        # tests / callers without a store only exercise registration shape
        scheduler.add_job(lambda: None, **kwargs)
        return
    scheduler.add_job(make_fire_fn(store, wf_id), **kwargs)


def remove_workflow_job(scheduler: Any, workflow_id: str) -> None:
    jid = f"{JOB_ID_PREFIX}{workflow_id}"
    if scheduler.get_job(jid) is not None:
        scheduler.remove_job(jid)


def sync_all_workflow_jobs(scheduler: Any, workflows: list[dict[str, Any]], store: Any | None = None) -> int:
    """Startup catch-up sweep: reconcile every workflow definition into jobs.

    Returns the number of scheduled (cron) workflows registered.
    """
    registered = 0
    for wf in workflows:
        before = len(getattr(scheduler, "jobs", {}))
        sync_workflow_job(scheduler, wf, store=store)
        after = len(getattr(scheduler, "jobs", {}))
        if after > before:
            registered += 1
    if workflows and store is not None:
        logger.info("workflow schedule sync: %d cron job(s) active", registered)
    return registered
