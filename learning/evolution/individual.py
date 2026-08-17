"""Weight-individual genotype for evolutionary search over decision weights."""
import random
from dataclasses import dataclass, field

_DIMENSIONS = ["policy_score", "outcome_score", "risk_score", "context_fit_score"]

_MIN_WEIGHT = 0.05
_MUTATION_STEP = 0.1


@dataclass
class WeightIndividual:
    """A weight config (per intent class) that sums to 1.0 with each dim >= 0.05."""

    intent_class: str
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.weights.keys()) != set(_DIMENSIONS):
            raise ValueError(f"weights must cover exactly {_DIMENSIONS}")
        if any(v < _MIN_WEIGHT for v in self.weights.values()):
            raise ValueError(f"each weight must be >= {_MIN_WEIGHT}")
        if abs(sum(self.weights.values()) - 1.0) > 0.001:
            raise ValueError(f"weights must sum to 1.0, got {sum(self.weights.values())}")

    def mutate(self, step: float = _MUTATION_STEP, rng: random.Random | None = None) -> "WeightIndividual":
        rng = rng or random.Random()
        weights = dict(self.weights)
        dim = rng.choice(_DIMENSIONS)
        others = [d for d in _DIMENSIONS if d != dim]

        # Bound the positive delta so the other dimensions can absorb it
        # (they must stay >= _MIN_WEIGHT).
        max_gain = sum(weights[d] - _MIN_WEIGHT for d in others)
        delta = rng.uniform(-step, step)
        delta = min(delta, max_gain)

        new_value = max(_MIN_WEIGHT, weights[dim] + delta)
        delta = new_value - weights[dim]
        weights[dim] = round(new_value, 4)

        # Redistribute so the sum stays 1.0. `remaining` is how much the OTHER
        # dims must change: negative (dim grew) → others shrink; positive
        # (dim shrank) → others absorb the freed weight.
        remaining = -delta
        if remaining < 0:
            # dim grew → others must shrink
            for d in sorted(others, key=lambda d: weights[d], reverse=True):
                take = min(-remaining, weights[d] - _MIN_WEIGHT)
                weights[d] = round(weights[d] - take, 4)
                remaining += take
        elif remaining > 0:
            # dim shrank → others absorb the freed weight
            for d in sorted(others, key=lambda d: weights[d]):
                give = min(remaining, 1.0 - weights[d])
                weights[d] = round(weights[d] + give, 4)
                remaining -= give

        # Hard clamp + renormalize: rounding across dims can leave a small
        # residual, so pin every value and rebalance the largest dim exactly.
        for d in weights:
            weights[d] = round(max(_MIN_WEIGHT, weights[d]), 4)
        total = sum(weights.values())
        diff = round(total - 1.0, 4)
        if diff > 0:
            for d in sorted(weights, key=lambda d: weights[d], reverse=True):
                take = min(diff, weights[d] - _MIN_WEIGHT)
                weights[d] = round(weights[d] - take, 4)
                diff = round(diff - take, 4)
                if diff <= 0:
                    break
        elif diff < 0:
            weights[max(weights, key=lambda d: weights[d])] = round(
                weights[max(weights, key=lambda d: weights[d])] - diff, 4
            )

        return WeightIndividual(intent_class=self.intent_class, weights=weights)

    def to_dict(self) -> dict:
        return {"intent_class": self.intent_class, "weights": self.weights}


def random_individual(intent_class: str, rng: random.Random | None = None) -> WeightIndividual:
    rng = rng or random.Random()
    raw = [rng.uniform(_MIN_WEIGHT, 1.0) for _ in _DIMENSIONS]
    total = sum(raw)
    normalized = [round(v / total, 4) for v in raw]

    # Enforce minimums after normalization.
    for i in range(len(normalized)):
        if normalized[i] < _MIN_WEIGHT:
            normalized[i] = _MIN_WEIGHT
    total = sum(normalized)
    if abs(total - 1.0) > 1e-6:
        diff = total - 1.0
        for i in range(len(normalized)):
            take = min(diff, normalized[i] - _MIN_WEIGHT)
            normalized[i] = round(normalized[i] - take, 4)
            diff -= take
            if diff <= 0:
                break

    return WeightIndividual(
        intent_class=intent_class,
        weights=dict(zip(_DIMENSIONS, normalized)),
    )
