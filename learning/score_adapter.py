"""Score Adapter — monitors dimension prediction accuracy; proposes weight adjustments."""
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import aiosqlite

from ..config import settings

logger = logging.getLogger(__name__)

_DIMENSIONS = ["policy_score", "outcome_score", "risk_score", "context_fit_score"]


class ScoreAdapter:
    def __init__(self, db_path: Path) -> None:
        self._db = db_path

    async def evaluate(self) -> list[dict]:
        """Check dimension accuracy. Returns list of proposed weight adjustments."""
        episodes = await self._fetch_episodes_with_scores()
        if not episodes:
            return []

        proposals = []
        for dim in _DIMENSIONS:
            predicted = [ep[dim] for ep in episodes if ep.get(dim) is not None]
            actuals = [1.0 if ep["outcome"] == "SUCCESS" else 0.0
                       for ep in episodes if ep.get(dim) is not None]
            if len(predicted) < 10:
                continue

            accuracy = self._correlation(predicted, actuals)
            if accuracy < settings.score_adapter_accuracy_threshold:
                logger.warning("Dimension %s accuracy %.2f < %.2f — proposing adjustment",
                               dim, accuracy, settings.score_adapter_accuracy_threshold)
                proposals.append({
                    "dimension": dim,
                    "current_accuracy": round(accuracy, 4),
                    "threshold": settings.score_adapter_accuracy_threshold,
                    "episode_count": len(predicted),
                })

        return proposals

    async def propose_weight_adjustment(
        self,
        intent_class: str,
        current_weights: dict,
        dimension: str,
        accuracy: float,
        episode_batch: str,
    ) -> str:
        """Write a pending weight config for operator review."""
        rate = settings.score_adapter_accuracy_threshold - accuracy
        adjustment = min(rate * 0.1, 0.05)

        new_weights = dict(current_weights)
        new_weights[dimension] = round(new_weights.get(dimension, 0.25) - adjustment, 4)

        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.001:
            other_dims = [d for d in _DIMENSIONS if d != dimension]
            redistribute = (total - 1.0) / len(other_dims)
            for d in other_dims:
                new_weights[d] = round(new_weights[d] - redistribute, 4)

        if any(v < 0.05 for v in new_weights.values()):
            logger.warning("Weight adjustment would violate minimum 0.05 constraint — skipping")
            return ""

        version = f"wc-proposed-{uuid4().hex[:8]}"
        async with aiosqlite.connect(self._db) as db:
            await db.execute(
                """INSERT INTO pending_weight_configs
                   (version, intent_class, weights, episode_batch, proposed_at, proposed_by)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (version, intent_class, json.dumps(new_weights), episode_batch,
                 time.time(), "score_adapter"),
            )
            await db.commit()
        return version

    def _correlation(self, predicted: list[float], actual: list[float]) -> float:
        n = len(predicted)
        if n < 2:
            return 1.0
        mean_p = sum(predicted) / n
        mean_a = sum(actual) / n
        num = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted, actual))
        den_p = (sum((p - mean_p) ** 2 for p in predicted)) ** 0.5
        den_a = (sum((a - mean_a) ** 2 for a in actual)) ** 0.5
        if den_p * den_a == 0:
            return 1.0
        return num / (den_p * den_a)

    async def _fetch_episodes_with_scores(self) -> list[dict]:
        # v0 stub: returns empty — full implementation queries episodes joined with decision scores
        return []
