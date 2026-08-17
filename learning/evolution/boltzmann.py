"""Adaptive-temperature Boltzmann selection.

Softmax over fitness scaled by temperature: cold temperature (annealed down
over generations) greedily prefers high-fitness individuals; hot temperature
keeps exploration high early.
"""
import math
import random


def boltzmann_softmax(fitnesses: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        temperature = 1e-9
    scaled = [f / temperature for f in fitnesses]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def select_parent_index(
    fitnesses: list[float],
    temperature: float,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random.Random()
    probs = boltzmann_softmax(fitnesses, temperature)
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(fitnesses) - 1


def annealing_temperature(initial: float, generation: int, decay: float) -> float:
    return initial * (decay ** generation)
