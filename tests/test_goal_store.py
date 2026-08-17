"""GoalStore — durable, resumable goal state: create, claim, and step advancement tests."""
import time

import aiosqlite
import pytest

from xnch.memory.db import init_db
from xnch.memory.goal_store import GoalStore


@pytest.fixture
async def db_path(tmp_path):
    path = tmp_path / "test.db"
    await init_db(path)
    return path


async def test_create_and_get(db_path):
    store = GoalStore(db_path)
    goal_id = await store.create_goal(
        owner_actor_id="actor-1",
        objective="Deploy the service",
        simulation_plan=[{"step": 1, "action": "deploy"}],
    )

    goal = await store.get_goal(goal_id)
    assert goal is not None
    assert goal["goal_id"] == goal_id
    assert goal["owner_actor_id"] == "actor-1"
    assert goal["objective"] == "Deploy the service"
    assert goal["status"] == "PENDING"
    assert goal["steps_completed"] == 0
    assert goal["consecutive_failures"] == 0
    assert goal["max_steps"] == 10
    assert goal["failure_threshold"] == 3
    # simulation_plan is stored as a JSON string, not read back parsed
    assert isinstance(goal["simulation_plan"], str)
    assert goal["simulation_plan"] == '[{"step": 1, "action": "deploy"}]'


async def test_claim_next_goal_marks_running(db_path):
    store = GoalStore(db_path)
    goal_id = await store.create_goal(owner_actor_id="actor-1", objective="Run task")

    claimed = await store.claim_next_goal(lease_owner="worker-1")
    assert claimed is not None
    assert claimed["goal_id"] == goal_id
    assert claimed["status"] == "RUNNING"
    assert claimed["lease_owner"] == "worker-1"
    assert claimed["lease_expires_at"] is not None

    goal = await store.get_goal(goal_id)
    assert goal["status"] == "RUNNING"
    assert goal["lease_owner"] == "worker-1"


async def test_claim_returns_none_when_none_eligible(db_path):
    store = GoalStore(db_path)
    assert await store.claim_next_goal(lease_owner="worker-1") is None


async def test_complete_step_success_increments(db_path):
    store = GoalStore(db_path)
    goal_id = await store.create_goal(owner_actor_id="actor-1", objective="Task", max_steps=5)
    await store.claim_next_goal(lease_owner="worker-1")

    updated = await store.complete_step(goal_id, "SUCCESS")
    assert updated is not None
    assert updated["steps_completed"] == 1
    assert updated["status"] == "ACTIVE"  # steps < max_steps (5)
    assert updated["consecutive_failures"] == 0
    assert updated["last_step_outcome"] == "SUCCESS"
    assert updated["lease_owner"] is None  # lease released


async def test_complete_step_success_completes(db_path):
    store = GoalStore(db_path)
    goal_id = await store.create_goal(owner_actor_id="actor-1", objective="Task", max_steps=1)
    await store.claim_next_goal(lease_owner="worker-1")

    updated = await store.complete_step(goal_id, "SUCCESS")
    assert updated is not None
    assert updated["steps_completed"] == 1
    assert updated["status"] == "COMPLETED"  # steps >= max_steps (1)
    assert updated["next_due_at"] is None  # terminal


async def test_complete_step_failure_threshold(db_path):
    store = GoalStore(db_path)
    goal_id = await store.create_goal(
        owner_actor_id="actor-1", objective="Task", failure_threshold=3
    )
    await store.claim_next_goal(lease_owner="worker-1")

    for _ in range(2):
        await store.complete_step(goal_id, "FAILURE")
    after_two = await store.get_goal(goal_id)
    assert after_two["consecutive_failures"] == 2
    assert after_two["status"] == "ACTIVE"  # below threshold

    failed = await store.complete_step(goal_id, "FAILURE")
    assert failed is not None
    assert failed["consecutive_failures"] == 3
    assert failed["status"] == "FAILED"  # threshold reached
    assert failed["next_due_at"] is None


async def test_claim_skips_running_and_terminal(db_path):
    store = GoalStore(db_path)
    g1 = await store.create_goal(owner_actor_id="actor-1", objective="first")
    g2 = await store.create_goal(owner_actor_id="actor-1", objective="second", max_steps=1)

    claimed = await store.claim_next_goal(lease_owner="worker-1")
    assert claimed is not None and claimed["goal_id"] == g1  # earliest due

    done = await store.complete_step(g2, "SUCCESS")
    assert done["status"] == "COMPLETED"  # terminal

    # g1 RUNNING (unexpired), g2 COMPLETED -> nothing eligible
    assert await store.claim_next_goal(lease_owner="worker-2") is None


async def test_claim_reclaims_expired_lease(db_path):
    store = GoalStore(db_path)
    goal_id = await store.create_goal(owner_actor_id="actor-1", objective="Task")
    await store.claim_next_goal(lease_owner="worker-1")

    # force the lease into the past
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE goals SET lease_expires_at = ? WHERE goal_id = ?",
            (time.time() - 10, goal_id),
        )
        await db.commit()

    claimed = await store.claim_next_goal(lease_owner="worker-2")
    assert claimed is not None
    assert claimed["goal_id"] == goal_id
    assert claimed["lease_owner"] == "worker-2"
    assert claimed["status"] == "RUNNING"
