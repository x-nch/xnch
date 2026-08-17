"""Per-intent-class island evolution.

Each island evolves its own population (island per intent class) using
Boltzmann parent selection with annealing temperature, and feeds the MAP-Elites
archive per island.
"""
import random

from .boltzmann import annealing_temperature, select_parent_index
from .individual import WeightIndividual, random_individual
from .map_elites import MapElitesArchive

_INITIAL_TEMPERATURE = 1.0
_TEMPERATURE_DECAY = 0.9


async def evolve_island(
    intent_class: str,
    fitness_fn,
    population_size: int = 20,
    generations: int = 10,
    rng: random.Random | None = None,
) -> tuple[WeightIndividual, float] | None:
    """Run one island's evolution; return the MAP-Elites archive best."""
    rng = rng or random.Random()

    population = [random_individual(intent_class, rng) for _ in range(population_size)]
    fitnesses = [await fitness_fn(ind) for ind in population]

    archive = MapElitesArchive()
    for ind, fit in zip(population, fitnesses):
        archive.add(ind, fit)

    for gen in range(generations):
        temperature = annealing_temperature(_INITIAL_TEMPERATURE, gen, _TEMPERATURE_DECAY)
        child = population[select_parent_index(fitnesses, temperature, rng)].mutate(rng=rng)
        child_fitness = await fitness_fn(child)
        archive.add(child, child_fitness)

        # Simple generational replacement: keep population diverse by
        # replacing the lowest-fitness member when the child improves on it.
        min_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if child_fitness > fitnesses[min_idx]:
            population[min_idx] = child
            fitnesses[min_idx] = child_fitness

    return archive.best()
