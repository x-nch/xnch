"""Memory stores — episodic + pattern round-trip tests."""
import pytest

from xnch.memory.db import init_db
from xnch.memory.episodic_store import EpisodicStore
from xnch.memory.pattern_store import PatternStore
from xnch.learning.pattern_extractor import PatternExtractor


@pytest.fixture
async def db_path(tmp_path):
    path = tmp_path / "test.db"
    await init_db(path)
    return path


@pytest.fixture
def episodic(db_path):
    return EpisodicStore(db_path)


@pytest.fixture
def patterns(db_path):
    return PatternStore(db_path)


async def test_create_and_complete_episode(episodic):
    episode_id = await episodic.create_episode(
        decision_id="dec-001",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="OPERATOR",
        context_snapshot={"test": True},
    )
    assert episode_id

    completed = await episodic.complete_episode(
        decision_id="dec-001",
        outcome="SUCCESS",
        observed_state_delta={"pod": "running"},
        side_effects=["ingress_registered"],
        duration_ms=5000,
        anomalies=[],
    )
    assert completed == episode_id


async def test_prediction_update(episodic):
    episode_id = await episodic.create_episode(
        "dec-002", "EXECUTION", "DEPLOY", "ML_MODEL", "OPERATOR", {}
    )
    await episodic.complete_episode("dec-002", "FAILURE", {}, [], 1000, [])
    await episodic.write_prediction_update(episode_id, 0.6, True)

    flagged = await episodic.get_flagged_for_early_extraction()
    assert ("EXECUTION", "DEPLOY", "ML_MODEL", "OPERATOR") in flagged


async def test_fetch_for_manifest(episodic):
    for i in range(3):
        await episodic.create_episode(
            f"dec-{i+10}", "QUERY", "LIST", "SERVICE", "VIEWER", {}
        )
        await episodic.complete_episode(f"dec-{i+10}", "SUCCESS", {}, [], 100, [])

    rows = await episodic.fetch_for_manifest("QUERY", "SERVICE", "VIEWER")
    assert len(rows) == 3


async def test_pattern_upsert_and_fetch(patterns):
    await patterns.upsert_pattern(
        context_signature="sha256:abc",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="OPERATOR",
        success_rate=0.75,
        confidence=0.61,
        observation_count=15,
        avg_prediction_delta=0.22,
        extraction_run_id="run-001",
    )

    results = await patterns.fetch_for_manifest("EXECUTION", "SERVICE", "OPERATOR")
    assert len(results) == 1
    assert results[0]["success_rate"] == 0.75
    assert results[0]["observation_count"] == 15


async def test_pattern_extractor_requires_min_observations(episodic, patterns):
    """Extractor must not write pattern until 10+ episodes exist."""
    for i in range(5):
        await episodic.create_episode(
            f"dec-p{i}", "DECISION", "PLAN", "SERVICE", "OPERATOR", {}
        )
        await episodic.complete_episode(f"dec-p{i}", "SUCCESS", {}, [], 100, [])

    extractor = PatternExtractor(episodic, patterns)
    written = await extractor.run()
    assert written == 0  # only 5 episodes, need 10


async def test_pattern_extractor_writes_at_threshold(episodic, patterns):
    for i in range(12):
        outcome = "SUCCESS" if i % 3 != 0 else "FAILURE"
        await episodic.create_episode(
            f"dec-q{i}", "QUERY", "READ_FILE", "FILE", "VIEWER", {}
        )
        await episodic.complete_episode(f"dec-q{i}", outcome, {}, [], 100, [])

    extractor = PatternExtractor(episodic, patterns)
    written = await extractor.run()
    assert written >= 1

    results = await patterns.fetch_for_manifest("QUERY", "FILE", "VIEWER")
    assert len(results) == 1
    assert 0.0 < results[0]["success_rate"] < 1.0
