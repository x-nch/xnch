"""MAP-Elites archive: population archive keyed by behavior descriptor.

Each cell holds the best individual (by fitness) for a discretized behavior
descriptor. Here the descriptor buckets the risk_score dimension, preserving
diversity across low- and high-risk weight configs while optimizing within
each cell.
"""
from typing import Any

from .individual import WeightIndividual

_NUM_RISK_BUCKETS = 4


def behavior_descriptor(ind: WeightIndividual) -> tuple[str, int]:
    """Bucket the risk dimension into a coarse cell key."""
    risk = ind.weights["risk_score"]
    bucket = min(int(risk * _NUM_RISK_BUCKETS), _NUM_RISK_BUCKETS - 1)
    return (ind.intent_class, bucket)


class MapElitesArchive:
    def __init__(self, descriptor_fn=None) -> None:
        self._descriptor_fn = descriptor_fn or behavior_descriptor
        self._cells: dict[tuple, tuple[WeightIndividual, float]] = {}

    def add(self, ind: Any, fitness: float) -> None:
        key = self._descriptor_fn(ind)
        current = self._cells.get(key)
        if current is None or fitness > current[1]:
            self._cells[key] = (ind, fitness)

    def get(self, key: tuple) -> Any | None:
        entry = self._cells.get(key)
        return entry[0] if entry else None

    def fitness(self, key: tuple) -> float | None:
        entry = self._cells.get(key)
        return entry[1] if entry else None

    def cells(self) -> list[tuple]:
        return list(self._cells.keys())

    def best(self) -> tuple[Any, float] | None:
        if not self._cells:
            return None
        return max(self._cells.values(), key=lambda e: e[1])

    def all_individuals(self) -> list[tuple[Any, float]]:
        return list(self._cells.values())
