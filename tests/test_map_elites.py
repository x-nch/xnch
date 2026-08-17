"""MAP-Elites behavior-descriptor archive tests."""
from xnch.learning.evolution.individual import WeightIndividual
from xnch.learning.evolution.map_elites import (
    behavior_descriptor,
    MapElitesArchive,
)


def _individual(**weights):
    base = {
        "policy_score": 0.25,
        "outcome_score": 0.30,
        "risk_score": 0.35,
        "context_fit_score": 0.10,
    }
    base.update(weights)
    # Renormalize to sum 1.0 while keeping all >= 0.05
    total = sum(base.values())
    normed = {k: round(v / total, 4) for k, v in base.items()}
    for k in normed:
        if normed[k] < 0.05:
            normed[k] = 0.05
    diff = sum(normed.values()) - 1.0
    if diff > 1e-6:
        for k in normed:
            take = min(diff, normed[k] - 0.05)
            normed[k] = round(normed[k] - take, 4)
            diff -= take
            if diff <= 0:
                break
    return WeightIndividual(intent_class="EXECUTION", weights=normed)


def test_behavior_descriptor_buckets_risk():
    low = _individual(risk_score=0.10)
    high = _individual(risk_score=0.90)
    assert behavior_descriptor(low) != behavior_descriptor(high)


def test_behavior_descriptor_stable_across_similar():
    a = _individual(risk_score=0.30)
    b = _individual(risk_score=0.34)
    assert behavior_descriptor(a) == behavior_descriptor(b)


def test_archive_adds_best_per_cell():
    archive = MapElitesArchive()
    low_fit = _individual(risk_score=0.10)
    high_fit = _individual(risk_score=0.10)

    archive.add(low_fit, fitness=0.4)
    archive.add(high_fit, fitness=0.9)

    key = behavior_descriptor(high_fit)
    assert archive.get(key) is high_fit
    assert archive.fitness(key) == 0.9


def test_archive_keeps_first_when_equal_fitness():
    archive = MapElitesArchive()
    a = _individual(risk_score=0.10)
    b = _individual(risk_score=0.10)
    archive.add(a, fitness=0.5)
    archive.add(b, fitness=0.5)
    key = behavior_descriptor(a)
    assert archive.get(key) is a


def test_archive_separates_cells():
    archive = MapElitesArchive()
    low = _individual(risk_score=0.10)
    high = _individual(risk_score=0.90)
    archive.add(low, fitness=0.5)
    archive.add(high, fitness=0.3)
    assert len(archive.cells()) == 2


def test_archive_best_overall():
    archive = MapElitesArchive()
    archive.add(_individual(risk_score=0.10), fitness=0.5)
    archive.add(_individual(risk_score=0.90), fitness=0.9)
    best = archive.best()
    assert best[1] == 0.9
