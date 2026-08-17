"""Fitness — prediction accuracy of a weight config over episodes tests."""
import json

from xnch.learning.evolution.fitness import compute_fitness, _correlation


def _episode(outcome, scores):
    return {
        "outcome": outcome,
        "scores_json": json.dumps(scores),
    }


def test_correlation_perfect_positive():
    assert abs(_correlation([0.1, 0.9], [0.0, 1.0]) - 1.0) < 1e-6


def test_correlation_perfect_negative():
    assert abs(_correlation([0.9, 0.1], [0.0, 1.0]) - (-1.0)) < 1e-6


def test_correlation_zero_variance_returns_neutral():
    assert _correlation([0.5, 0.5], [0.0, 1.0]) == 0.0


def test_compute_fitness_prefers_weights_matching_outcomes():
    # policy_score strongly tracks SUCCESS; risk_score tracks FAILURE.
    episodes = [
        _episode("SUCCESS", {"policy_score": 0.9, "outcome_score": 0.5, "risk_score": 0.2, "context_fit_score": 0.5}),
        _episode("SUCCESS", {"policy_score": 0.8, "outcome_score": 0.5, "risk_score": 0.3, "context_fit_score": 0.5}),
        _episode("SUCCESS", {"policy_score": 0.85, "outcome_score": 0.5, "risk_score": 0.25, "context_fit_score": 0.5}),
        _episode("FAILURE", {"policy_score": 0.2, "outcome_score": 0.5, "risk_score": 0.8, "context_fit_score": 0.5}),
        _episode("FAILURE", {"policy_score": 0.1, "outcome_score": 0.5, "risk_score": 0.9, "context_fit_score": 0.5}),
    ]
    # Weight config emphasizing policy_score (the success-tracking dim) → high fitness
    policy_heavy = {
        "policy_score": 0.7, "outcome_score": 0.1,
        "risk_score": 0.1, "context_fit_score": 0.1,
    }
    fitness = compute_fitness(policy_heavy, episodes)
    assert fitness > 0.5

    # Risk-heavy weights inflate composite for failures → low fitness
    risk_heavy = {
        "policy_score": 0.1, "outcome_score": 0.1,
        "risk_score": 0.7, "context_fit_score": 0.1,
    }
    assert compute_fitness(risk_heavy, episodes) < 0.5


def test_compute_fitness_needs_min_episodes():
    episodes = [_episode("SUCCESS", {"policy_score": 0.9, "outcome_score": 0.8, "risk_score": 0.2, "context_fit_score": 0.9})]
    fitness = compute_fitness({}, episodes)
    assert fitness == 0.0


def test_compute_fitness_filters_missing_scores():
    weights = {
        "policy_score": 0.25, "outcome_score": 0.30,
        "risk_score": 0.35, "context_fit_score": 0.10,
    }
    episodes = [
        _episode("SUCCESS", {"policy_score": 0.9, "outcome_score": 0.8, "risk_score": 0.2, "context_fit_score": 0.9}),
        _episode("SUCCESS", {"policy_score": 0.85, "outcome_score": 0.75, "risk_score": 0.25, "context_fit_score": 0.85}),
        _episode("SUCCESS", {"policy_score": 0.8, "outcome_score": 0.7, "risk_score": 0.3, "context_fit_score": 0.8}),
        _episode("SUCCESS", {}),  # missing scores → filtered out
        _episode("FAILURE", {"policy_score": 0.1, "outcome_score": 0.2, "risk_score": 0.9, "context_fit_score": 0.1}),
        _episode("FAILURE", {"policy_score": 0.15, "outcome_score": 0.25, "risk_score": 0.85, "context_fit_score": 0.15}),
    ]
    fitness = compute_fitness(weights, episodes)
    assert fitness > 0.0
