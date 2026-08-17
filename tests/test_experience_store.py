"""ExperienceStore — experiential memory upsert + manifest retrieval tests."""
from uuid import uuid4

import pytest

from xnch.memory.db import init_db
from xnch.memory.experience_store import ExperienceStore
from xnch.memory.pattern_store import PatternStore


@pytest.fixture
async def db_path(tmp_path):
    path = tmp_path / "test.db"
    await init_db(path)
    return path


async def test_upsert_and_fetch(db_path):
    store = ExperienceStore(db_path)
    await store.upsert_experience(
        context_signature="sha256:abc",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="OPERATOR",
        outcome="FAILURE",
        lesson="Rollback first, then stage",
        insight="Staging directly caused outage",
        verdict="MODIFY",
        applicability="EXECUTION|DEPLOY|SERVICE",
    )

    rows = await store.fetch_for_manifest("EXECUTION", "SERVICE", "OPERATOR")
    assert len(rows) == 1
    assert rows[0]["lesson"] == "Rollback first, then stage"
    assert rows[0]["verdict"] == "MODIFY"
    assert rows[0]["confidence"] == 0.6667  # Beta(2,1) after 1 observation


async def test_upsert_same_signature_updates_confidence(db_path):
    store = ExperienceStore(db_path)
    await store.upsert_experience(
        context_signature="sha256:abc",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="OPERATOR",
        outcome="FAILURE",
        lesson="Lesson one",
        insight="Insight one",
        verdict="MODIFY",
        applicability="EXECUTION|DEPLOY|SERVICE",
    )
    await store.upsert_experience(
        context_signature="sha256:abc",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="OPERATOR",
        outcome="FAILURE",
        lesson="Lesson two",
        insight="Insight two",
        verdict="BLOCK",
        applicability="EXECUTION|DEPLOY|SERVICE",
    )

    rows = await store.fetch_for_manifest("EXECUTION", "SERVICE", "OPERATOR")
    assert len(rows) == 1
    assert rows[0]["observation_count"] == 2
    # Beta(3,1) → 0.75
    assert rows[0]["confidence"] == 0.75


async def test_fetch_scoped_by_intent_entity_actor(db_path):
    store = ExperienceStore(db_path)
    await store.upsert_experience(
        context_signature="sha256:abc",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="OPERATOR",
        outcome="FAILURE",
        lesson="A",
        insight="A",
        verdict="MODIFY",
        applicability="EXECUTION|DEPLOY|SERVICE",
    )
    await store.upsert_experience(
        context_signature="sha256:def",
        intent_class="QUERY",
        action_type="LIST",
        entity_class="FILE",
        actor_role="VIEWER",
        outcome="SUCCESS",
        lesson="B",
        insight="B",
        verdict="ALLOW",
        applicability="QUERY|LIST|FILE",
    )

    rows = await store.fetch_for_manifest("QUERY", "FILE", "VIEWER")
    assert len(rows) == 1
    assert rows[0]["lesson"] == "B"


async def test_fetch_returns_empty_when_no_match(db_path):
    store = ExperienceStore(db_path)
    rows = await store.fetch_for_manifest("EXECUTION", "SERVICE", "OPERATOR")
    assert rows == []


async def test_store_uses_same_db_as_patterns(db_path):
    """ExperienceStore and PatternStore must share the SQLite file safely."""
    exp = ExperienceStore(db_path)
    pat = PatternStore(db_path)
    await exp.upsert_experience(
        context_signature="sha256:abc",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="OPERATOR",
        outcome="FAILURE",
        lesson="A",
        insight="A",
        verdict="MODIFY",
        applicability="EXECUTION|DEPLOY|SERVICE",
    )
    await pat.upsert_pattern(
        context_signature="sha256:abc",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="OPERATOR",
        success_rate=0.5,
        confidence=0.6,
        observation_count=3,
        avg_prediction_delta=0.1,
        extraction_run_id="run-1",
    )
    assert await exp.fetch_for_manifest("EXECUTION", "SERVICE", "OPERATOR")
    assert await pat.fetch_for_manifest("EXECUTION", "SERVICE", "OPERATOR")
