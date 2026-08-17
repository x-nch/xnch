"""Weight-individual genotype + mutation tests (evolution package)."""
import pytest

from xnch.learning.evolution.individual import (
    WeightIndividual,
    random_individual,
    _DIMENSIONS,
)


def test_individual_from_weights_normalizes():
    ind = WeightIndividual(intent_class="EXECUTION", weights={
        "policy_score": 0.5, "outcome_score": 0.3,
        "risk_score": 0.15, "context_fit_score": 0.05,
    })
    assert ind.intent_class == "EXECUTION"
    assert abs(sum(ind.weights.values()) - 1.0) < 0.001
    assert all(v >= 0.05 for v in ind.weights.values())
    assert set(ind.weights.keys()) == set(_DIMENSIONS)


def test_individual_rejects_invalid_weights():
    with pytest.raises(ValueError):
        WeightIndividual(intent_class="EXECUTION", weights={
            "policy_score": 0.9, "outcome_score": 0.3,
            "risk_score": 0.2, "context_fit_score": 0.1,
        })  # sums to 1.5


def test_individual_rejects_weight_below_minimum():
    with pytest.raises(ValueError):
        WeightIndividual(intent_class="EXECUTION", weights={
            "policy_score": 0.7, "outcome_score": 0.2,
            "risk_score": 0.05, "context_fit_score": 0.04,
        })


def test_mutation_keeps_validity():
    ind = WeightIndividual(intent_class="EXECUTION", weights={
        "policy_score": 0.25, "outcome_score": 0.30,
        "risk_score": 0.35, "context_fit_score": 0.10,
    })
    for _ in range(20):
        child = ind.mutate(step=0.1)
        assert child.intent_class == "EXECUTION"
        assert abs(sum(child.weights.values()) - 1.0) < 0.001
        assert all(v >= 0.05 for v in child.weights.values())


def test_mutation_changes_weights():
    ind = WeightIndividual(intent_class="EXECUTION", weights={
        "policy_score": 0.25, "outcome_score": 0.30,
        "risk_score": 0.35, "context_fit_score": 0.10,
    })
    changed = sum(1 for _ in range(50)
                  if ind.mutate(step=0.2).weights != ind.weights)
    assert changed > 0


def test_mutation_never_crashes_on_boundary_weights():
    """Regression: mutate must not raise ValueError when dims sit near the minimum.

    Reproduces a sum-drift crash (weights summing to 1.0075) when a dimension
    near the 0.05 floor gets a large delta and others lack slack to absorb it.
    """
    import random

    for seed in range(500):
        rng = random.Random(seed)
        ind = random_individual("EXECUTION", rng)
        for _ in range(50):
            ind = ind.mutate(step=0.3, rng=rng)
            assert abs(sum(ind.weights.values()) - 1.0) < 0.001
            assert all(v >= 0.05 for v in ind.weights.values())


def test_random_individual_is_valid():
    for _ in range(20):
        ind = random_individual("QUERY")
        assert ind.intent_class == "QUERY"
        assert abs(sum(ind.weights.values()) - 1.0) < 0.001
        assert all(v >= 0.05 for v in ind.weights.values())


def test_as_dict_shape():
    ind = WeightIndividual(intent_class="EXECUTION", weights={
        "policy_score": 0.25, "outcome_score": 0.30,
        "risk_score": 0.35, "context_fit_score": 0.10,
    })
    d = ind.to_dict()
    assert d["intent_class"] == "EXECUTION"
    assert d["weights"] == ind.weights
