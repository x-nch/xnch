"""WeightEvolver orchestrator tests."""
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from xnch.learning.evolution.evolver import WeightEvolver


def _episode(intent_class, outcome, policy, risk, created_days_ago=0):
    return {
        "episode_id": f"ep-{policy}-{risk}",
        "decision_id": f"dec-{policy}-{risk}",
        "intent_class": intent_class,
        "action_type": "DEPLOY",
        "entity_class": "SERVICE",
        "actor_role": "operator",
        "outcome": outcome,
        "prediction_delta": 0.1,
        "scores_json": json.dumps({
            "policy_score": policy, "outcome_score": 0.5,
            "risk_score": risk, "context_fit_score": 0.5,
        }),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }


def _episode_batch():
    """Episodes where policy_score strongly tracks success (evolver should find it)."""
    return [
        _episode("EXECUTION", "SUCCESS", 0.9, 0.2),
        _episode("EXECUTION", "SUCCESS", 0.85, 0.25),
        _episode("EXECUTION", "SUCCESS", 0.8, 0.3),
        _episode("EXECUTION", "SUCCESS", 0.88, 0.22),
        _episode("EXECUTION", "SUCCESS", 0.82, 0.28),
        _episode("EXECUTION", "FAILURE", 0.1, 0.9),
        _episode("EXECUTION", "FAILURE", 0.2, 0.8),
        _episode("EXECUTION", "FAILURE", 0.15, 0.85),
        _episode("EXECUTION", "FAILURE", 0.12, 0.88),
        _episode("EXECUTION", "FAILURE", 0.18, 0.82),
    ]


async def test_evolver_proposes_winner_when_better():
    fetch = AsyncMock(return_value=_episode_batch())
    propose = AsyncMock(return_value="wc-proposed-abc")
    current = AsyncMock(return_value={
        "version": "default-v0",
        "weights": {
            "policy_score": 0.25, "outcome_score": 0.30,
            "risk_score": 0.35, "context_fit_score": 0.10,
        },
    })

    evolver = WeightEvolver(fetch_fn=fetch, propose_fn=propose, current_weights_fn=current)
    results = await evolver.run()

    assert len(results) == 1
    assert results[0]["intent_class"] == "EXECUTION"
    assert results[0]["version"] == "wc-proposed-abc"
    assert results[0]["winner_fitness"] > results[0]["current_fitness"]


async def test_evolver_skips_when_winner_not_better():
    fetch = AsyncMock(return_value=_episode_batch())
    propose = AsyncMock(return_value="wc-proposed-abc")
    # Current weights are already near-optimal → winner should not beat them
    current = AsyncMock(return_value={
        "version": "default-v0",
        "weights": {
            "policy_score": 0.7, "outcome_score": 0.1,
            "risk_score": 0.1, "context_fit_score": 0.1,
        },
    })

    evolver = WeightEvolver(fetch_fn=fetch, propose_fn=propose, current_weights_fn=current)
    results = await evolver.run()

    assert results == []
    propose.assert_not_awaited()


async def test_evolver_skips_intent_classes_without_enough_episodes():
    # Only 2 episodes → below the min needed for fitness
    fetch = AsyncMock(return_value=_episode_batch()[:2])
    propose = AsyncMock(return_value="wc-proposed-abc")
    current = AsyncMock(return_value={
        "version": "default-v0",
        "weights": {
            "policy_score": 0.25, "outcome_score": 0.30,
            "risk_score": 0.35, "context_fit_score": 0.10,
        },
    })

    evolver = WeightEvolver(fetch_fn=fetch, propose_fn=propose, current_weights_fn=current)
    results = await evolver.run()

    assert results == []
    propose.assert_not_awaited()


async def test_evolver_propose_payload_shape():
    fetch = AsyncMock(return_value=_episode_batch())
    propose = AsyncMock(return_value="wc-proposed-abc")
    current = AsyncMock(return_value={
        "version": "default-v0",
        "weights": {
            "policy_score": 0.25, "outcome_score": 0.30,
            "risk_score": 0.35, "context_fit_score": 0.10,
        },
    })

    evolver = WeightEvolver(fetch_fn=fetch, propose_fn=propose, current_weights_fn=current)
    await evolver.run()

    payload = propose.call_args.kwargs
    assert payload["intent_class"] == "EXECUTION"
    assert set(payload["weights"].keys()) == {"policy_score", "outcome_score", "risk_score", "context_fit_score"}
    assert abs(sum(payload["weights"].values()) - 1.0) < 0.001
    assert "episode_batch" in payload


async def test_evolver_fetch_min_fitness_for_generation_is_injected():
    """Generation threshold must be configurable via the constructor."""
    evolver = WeightEvolver(
        fetch_fn=AsyncMock(return_value=[]),
        propose_fn=AsyncMock(return_value=""),
        current_weights_fn=AsyncMock(return_value={}),
        min_fitness_gain=0.05,
    )
    assert evolver._min_fitness_gain == 0.05
