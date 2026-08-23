"""P3 — scheduled-workflow jobs on the existing AsyncIOScheduler.

Fails until xnch/jobs/workflow_schedule.py exists with:
  - job_kwargs_for_trigger(wf_row)   -> APScheduler kwargs or None (manual)
  - make_fire_fn(store, workflow_id) -> async callable creating a run
  - sync_workflow_job / remove_workflow_job / sync_all_workflow_jobs

Uses a fake scheduler object (add_job/remove_job/get_jobs contract), so no
real AsyncIOScheduler instance is needed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

XNCH_ROOT = Path(__file__).resolve().parent.parent
if str(XNCH_ROOT) not in sys.path:
    sys.path.insert(0, str(XNCH_ROOT))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, XNCH_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_db = _load("xnch_p3_db", "memory/db.py")
_store_mod = _load("xnch_p3_store", "memory/workflow_store.py")
init_db = _db.init_db
WorkflowStore = _store_mod.WorkflowStore


def _load_jobs_module():
    return _load("xnch_p3_jobs", "jobs/workflow_schedule.py")


class FakeJob:
    def __init__(self, id):
        self.id = id


class FakeScheduler:
    def __init__(self):
        self.jobs: dict[str, dict] = {}

    def add_job(self, fn, **kwargs):
        jid = kwargs["id"]
        self.jobs[jid] = {"fn": fn, **kwargs}

    def remove_job(self, jid):
        self.jobs.pop(jid, None)

    def get_job(self, jid):
        return FakeJob(jid) if jid in self.jobs else None


def _wf(id_, kind, cron=None):
    return {
        "id": id_,
        "name": "W",
        "trigger_json": (
            '{"kind":"schedule","cron":"%s"}' % cron
            if kind == "schedule"
            else '{"kind":"manual"}'
        ),
        "steps_json": "[]",
    }


# ----------------------------------------------------------------------


async def test_manual_trigger_registers_nothing(tmp_path):
    jobs = _load_jobs_module()
    db = tmp_path / "s.db"
    await init_db(db)
    sched = FakeScheduler()

    jobs.sync_workflow_job(sched, _wf("wf_m", "manual"))
    assert sched.jobs == {}


async def test_schedule_trigger_registers_cron_job_with_catchup(tmp_path):
    jobs = _load_jobs_module()
    sched = FakeScheduler()

    jobs.sync_workflow_job(sched, _wf("wf_1", "schedule", cron="0 9 * * 1"))

    assert "workflow:wf_1" in sched.jobs
    kwargs = sched.jobs["workflow:wf_1"]
    # restart catch-up semantics: missed fires within grace window still run
    assert kwargs["misfire_grace_time"] >= 3600
    assert kwargs["replace_existing"] is True
    # trigger is an APScheduler CronTrigger parsed from the 5-field expr
    from apscheduler.triggers.cron import CronTrigger

    assert isinstance(kwargs["trigger"], CronTrigger)


async def test_resync_replaces_existing_job(tmp_path):
    jobs = _load_jobs_module()
    sched = FakeScheduler()
    wf = _wf("wf_r", "schedule", cron="0 9 * * 1")

    jobs.sync_workflow_job(sched, wf)
    jobs.sync_workflow_job(sched, wf)  # e.g. PATCHed definition
    assert list(sched.jobs.keys()) == ["workflow:wf_r"]


async def test_remove_workflow_job(tmp_path):
    jobs = _load_jobs_module()
    sched = FakeScheduler()
    jobs.sync_workflow_job(sched, _wf("wf_d", "schedule", cron="*/5 * * * *"))

    jobs.remove_workflow_job(sched, "wf_d")
    assert "workflow:wf_d" not in sched.jobs
    # removing a manual/absent job must not raise
    jobs.remove_workflow_job(sched, "wf_absent")


async def test_sync_all_registers_only_scheduled(tmp_path):
    jobs = _load_jobs_module()
    sched = FakeScheduler()
    rows = [
        _wf("a", "manual"),
        _wf("b", "schedule", cron="0 0 * * *"),
        _wf("c", "schedule", cron="30 4 * * *"),
    ]
    jobs.sync_all_workflow_jobs(sched, rows)
    assert set(sched.jobs.keys()) == {"workflow:b", "workflow:c"}


async def test_fire_fn_creates_run_with_schedule_trigger(tmp_path):
    jobs = _load_jobs_module()
    db_path = tmp_path / "f.db"
    await init_db(db_path)
    store = WorkflowStore(db_path)
    wf_id = await store.create_workflow(
        owner_actor_id="system",
        name="Digest",
        description=None,
        trigger={"kind": "schedule", "cron": "0 9 * * 1"},
        steps=[{"id": "s", "kind": "exec_tool", "summary": "go",
                "requires_approval": True}],
    )

    fire = jobs.make_fire_fn(store, wf_id)
    await fire()

    runs = await store.list_runs(workflow_id=wf_id)
    assert len(runs) == 1
    import json

    trigger = json.loads(runs[0]["trigger_json"])
    assert trigger["kind"] == "schedule"


async def test_invalid_cron_is_skipped_not_fatal():
    jobs = _load_jobs_module()
    sched = FakeScheduler()
    # must not raise; job simply not registered
    jobs.sync_workflow_job(sched, _wf("bad", "schedule", cron="not-a-cron"))
    assert sched.jobs == {}


async def test_main_lifespan_registers_scheduled_workflows():
    """main.py wires sync_all_workflow_jobs into lifespan — static check."""
    src = (XNCH_ROOT / "main.py").read_text()
    assert "sync_all_workflow_jobs" in src
    assert "scheduler" in src
