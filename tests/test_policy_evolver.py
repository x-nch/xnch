"""PolicyRuleEvolver orchestrator tests."""
import json
import random
from unittest.mock import AsyncMock

from xnch.learning.evolution.policy_evolver import PolicyRuleEvolver
from xnch.learning.evolution.policy_rule import PolicyRule


def _seeded_evolver(**kwargs):
    kwargs.setdefault("rng", random.Random(42))
    return PolicyRuleEvolver(**kwargs)


def _episode(intent_class, action_type, entity_class, actor_role, outcome):
    return {
        "episode_id": f"ep-{outcome}-{action_type}",
        "intent_class": intent_class,
        "action_type": action_type,
        "entity_class": entity_class,
        "actor_role": actor_role,
        "outcome": outcome,
        "scores_json": json.dumps({"policy_score": 0.5, "outcome_score": 0.5,
                                   "risk_score": 0.5, "context_fit_score": 0.5}),
    }


def _episode_batch():
    """Episodes where DEPLOY on SERVICE fails but LIST on FILE succeeds."""
    eps = []
    for i in range(8):
        eps.append(_episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "FAILURE"))
    for i in range(8):
        eps.append(_episode("QUERY", "LIST", "FILE", "viewer", "SUCCESS"))
    return eps


async def test_evolver_proposes_rule_when_fitness_gain():
    store_write = AsyncMock(return_value="cand-1")
    evolver = _seeded_evolver(
        episodes_fn=AsyncMock(return_value=_episode_batch()),
        candidate_write_fn=store_write,
    )
    results = await evolver.run()

    assert len(results) >= 1
    for r in results:
        assert r["rule_yaml"]
        assert r["fitness"] > 0.5
    store_write.assert_called()


async def test_evolver_does_not_propose_neutral_rules():
    """No proposal when evolved rules can't beat the neutral baseline."""
    # Mixed episodes: nothing correlates → all rules land near 0.5.
    eps = []
    for i in range(6):
        eps.append(_episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "SUCCESS"))
        eps.append(_episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "FAILURE"))
        eps.append(_episode("QUERY", "LIST", "FILE", "viewer", "SUCCESS"))
        eps.append(_episode("QUERY", "LIST", "FILE", "viewer", "FAILURE"))

    store_write = AsyncMock(return_value="cand-1")
    evolver = _seeded_evolver(
        episodes_fn=AsyncMock(return_value=eps),
        candidate_write_fn=store_write,
        min_fitness=0.8,
    )
    results = await evolver.run()
    assert results == []
    store_write.assert_not_awaited()


async def test_evolver_skips_when_insufficient_episodes():
    evolver = _seeded_evolver(
        episodes_fn=AsyncMock(return_value=_episode_batch()[:3]),
        candidate_write_fn=AsyncMock(return_value="cand-1"),
    )
    results = await evolver.run()
    assert results == []


async def test_evolver_candidate_payload_shape():
    store_write = AsyncMock(return_value="cand-1")
    evolver = _seeded_evolver(
        episodes_fn=AsyncMock(return_value=_episode_batch()),
        candidate_write_fn=store_write,
    )
    await evolver.run()

    payload = store_write.call_args.kwargs
    assert "rule_yaml" in payload
    assert "triggering_pattern" in payload
    assert "candidate_id" in payload


async def test_evolver_batch_limits_proposals():
    """Evolver should not flood candidates — batch capped."""
    store_write = AsyncMock(return_value="cand-1")
    evolver = _seeded_evolver(
        episodes_fn=AsyncMock(return_value=_episode_batch()),
        candidate_write_fn=store_write,
        batch_size=2,
    )
    results = await evolver.run()
    assert len(results) <= 2
