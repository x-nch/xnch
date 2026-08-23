"""AgentRunStore unit tests: dispatch queue semantics on real SQLite.

Mirrors test_workflow_store.py conventions: tmp_path db, asyncio-auto tests.
Covers: create defaults, FIFO claiming with lease, lease exclusivity,
expired-lease re-claim, terminal outcomes, list filtering.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from xnch.memory.agent_run_store import AgentRunStore
from xnch.memory.db import init_db


@pytest.fixture()
async def store(tmp_path: Path) -> AgentRunStore:
    db = tmp_path / "xnch.db"
    await init_db(db)
    return AgentRunStore(db)


async def test_create_run_defaults_to_queued(store: AgentRunStore) -> None:
    row = await store.create_run(prompt="do a thing", workspace="~/xnch-agents/x")
    assert row["status"] == "QUEUED"
    assert row["prompt"] == "do a thing"
    assert row["workspace"] == "~/xnch-agents/x"
    assert row["runner_id"] is None
    assert row["lease_expires_at"] is None
    assert row["exit_code"] is None


async def test_claim_fifo_sets_running_with_lease(store: AgentRunStore) -> None:
    await store.create_run(prompt="first", workspace="w1")
    await store.create_run(prompt="second", workspace="w2")

    now = time.time()
    claimed = await store.claim_next("mac-runner", ttl_s=1800)
    assert claimed is not None
    assert claimed["prompt"] == "first"
    assert claimed["status"] == "RUNNING"
    assert claimed["runner_id"] == "mac-runner"
    assert claimed["lease_expires_at"] is not None
    assert claimed["lease_expires_at"] > now


async def test_claim_returns_none_when_only_unexpired_running(store: AgentRunStore) -> None:
    await store.create_run(prompt="only", workspace="w")
    await store.claim_next("r1", ttl_s=1800)
    assert await store.claim_next("r2", ttl_s=1800) is None


async def test_expired_lease_is_reclaimable(store: AgentRunStore) -> None:
    import aiosqlite

    row = await store.create_run(prompt="stalled", workspace="w")
    await store.claim_next("r1", ttl_s=1)
    # Force lease expiry without sleeping.
    async with aiosqlite.connect(store.db_path) as db:
        await db.execute("UPDATE agent_runs SET lease_expires_at = ? WHERE id = ?", (time.time() - 1, row["id"]))
        await db.commit()
    reclaimed = await store.claim_next("r2", ttl_s=1800)
    assert reclaimed is not None
    assert reclaimed["id"] == row["id"]
    assert reclaimed["runner_id"] == "r2"


async def test_complete_run_done_writes_fields(store: AgentRunStore) -> None:
    row = await store.create_run(prompt="p", workspace="w")
    await store.claim_next("r1", ttl_s=60)
    done = await store.complete_run(
        row["id"],
        outcome_status="DONE",
        exit_code=0,
        output_path=str(Path("/tmp/out")),
    )
    assert done is not None
    assert done["status"] == "DONE"
    assert done["exit_code"] == 0
    assert done["output_path"].endswith("out")
    assert done["error"] is None


async def test_complete_run_failed_carries_error(store: AgentRunStore) -> None:
    row = await store.create_run(prompt="p", workspace="w")
    await store.claim_next("r1", ttl_s=60)
    failed = await store.complete_run(row["id"], outcome_status="FAILED", error="boom")
    assert failed is not None
    assert failed["status"] == "FAILED"
    assert failed["error"] == "boom"


async def test_complete_on_queued_returns_none(store: AgentRunStore) -> None:
    row = await store.create_run(prompt="p", workspace="w")
    assert await store.complete_run(row["id"], outcome_status="DONE") is None


async def test_complete_unknown_run_returns_none(store: AgentRunStore) -> None:
    assert await store.complete_run("nope", outcome_status="DONE") is None


async def test_list_runs_filters_by_status(store: AgentRunStore) -> None:
    a = await store.create_run(prompt="a", workspace="w")
    b = await store.create_run(prompt="b", workspace="w")
    claimed = await store.claim_next("r1", ttl_s=60)
    assert claimed is not None and claimed["id"] == a["id"]
    await store.complete_run(a["id"], outcome_status="DONE")

    queued = await store.list_runs(status="QUEUED")
    running = await store.list_runs(status="RUNNING")
    done = await store.list_runs(status="DONE")
    assert [r["prompt"] for r in queued] == ["b"]
    assert [r["prompt"] for r in running] == []
    assert [r["prompt"] for r in done] == ["a"]
