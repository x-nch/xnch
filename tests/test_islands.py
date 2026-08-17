"""Per-intent-class island evolution tests."""
from unittest.mock import AsyncMock

from xnch.learning.evolution.individual import random_individual
from xnch.learning.evolution.islands import evolve_island


async def test_evolve_island_returns_archive_best():
    async def fitness_fn(ind):
        # Fitness favors low risk weight
        return 1.0 - ind.weights["risk_score"]

    best = await evolve_island(
        intent_class="EXECUTION",
        fitness_fn=fitness_fn,
        population_size=10,
        generations=5,
        rng=None,
    )

    assert best is not None
    assert best[0].intent_class == "EXECUTION"
    assert 0.0 <= best[1] <= 1.0


async def test_evolve_island_improves_over_random():
    async def fitness_fn(ind):
        return 1.0 - ind.weights["risk_score"]

    best = await evolve_island(
        intent_class="QUERY",
        fitness_fn=fitness_fn,
        population_size=10,
        generations=10,
        rng=None,
    )

    assert best[0].weights["risk_score"] <= 0.2


async def test_evolve_island_calls_fitness_on_each_individual():
    calls = []

    async def fitness_fn(ind):
        calls.append(ind.intent_class)
        return 0.5

    await evolve_island(
        intent_class="DECISION",
        fitness_fn=fitness_fn,
        population_size=4,
        generations=2,
        rng=None,
    )

    assert len(calls) >= 4
    assert all(c == "DECISION" for c in calls)
