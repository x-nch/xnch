"""Adaptive-temperature Boltzmann selection tests."""
import random

from xnch.learning.evolution.boltzmann import boltzmann_softmax, select_parent_index


def test_softmax_weights_high_fitness_individuals():
    fitnesses = [0.1, 0.9]
    probs = boltzmann_softmax(fitnesses, temperature=0.1)
    assert len(probs) == 2
    assert abs(sum(probs) - 1.0) < 1e-6
    assert probs[1] > probs[0]


def test_softmax_high_temperature_flattens():
    fitnesses = [0.1, 0.9]
    hot = boltzmann_softmax(fitnesses, temperature=100.0)
    cold = boltzmann_softmax(fitnesses, temperature=0.01)
    # Hot selection should be nearly uniform
    assert abs(hot[0] - hot[1]) < 0.1
    # Cold selection should strongly favor the fitter individual
    assert cold[1] > 0.9


def test_select_parent_index_prefers_fit():
    rng = random.Random(42)
    counts = {0: 0, 1: 0}
    for _ in range(200):
        idx = select_parent_index([0.1, 0.9], temperature=0.1, rng=rng)
        counts[idx] += 1
    assert counts[1] > counts[0]


def test_temperature_schedule_anneals():
    from xnch.learning.evolution.boltzmann import annealing_temperature
    temps = [annealing_temperature(initial=1.0, generation=g, decay=0.9) for g in range(10)]
    assert temps[0] == 1.0
    assert all(temps[i] > temps[i + 1] for i in range(len(temps) - 1))
