"""Fitness — how well a weight config predicts decision outcomes.

For each episode with per-dimension scores, compute the weighted composite
score under the candidate weights, then measure the correlation between that
composite and the actual binary outcome. Higher correlation = better weights.
"""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_MIN_EPISODES = 5
_DIMENSIONS = ["policy_score", "outcome_score", "risk_score", "context_fit_score"]


def _correlation(predicted: list[float], actual: list[float]) -> float:
    n = len(predicted)
    if n < 2:
        return 0.0
    mean_p = sum(predicted) / n
    mean_a = sum(actual) / n
    num = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted, actual))
    den_p = (sum((p - mean_p) ** 2 for p in predicted)) ** 0.5
    den_a = (sum((a - mean_a) ** 2 for a in actual)) ** 0.5
    if den_p * den_a == 0:
        return 0.0
    return num / (den_p * den_a)


def _parse_scores(episode: dict[str, Any]) -> dict[str, float] | None:
    raw = episode.get("scores_json")
    if not raw:
        return None
    try:
        scores = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(scores, dict):
        return None
    return {k: float(v) for k, v in scores.items() if k in _DIMENSIONS and v is not None}


def compute_fitness(weights: dict[str, float], episodes: list[dict[str, Any]]) -> float:
    """Correlation between weighted composite and binary outcome (0..1)."""
    composites: list[float] = []
    actuals: list[float] = []

    for ep in episodes:
        scores = _parse_scores(ep)
        if not scores:
            continue
        outcome = ep.get("outcome")
        if outcome not in ("SUCCESS", "FAILURE", "PARTIAL"):
            continue
        composite = sum(weights.get(dim, 0.0) * scores.get(dim, 0.0) for dim in _DIMENSIONS)
        if not weights:
            composite = sum(scores.get(dim, 0.0) for dim in _DIMENSIONS) / len(_DIMENSIONS)
        composites.append(composite)
        actuals.append(1.0 if outcome == "SUCCESS" else 0.0)

    if len(composites) < _MIN_EPISODES:
        return 0.0

    corr = _correlation(composites, actuals)
    # Map [-1, 1] correlation to [0, 1] fitness.
    return max(0.0, (corr + 1.0) / 2.0)
