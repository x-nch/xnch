"""Score Adapter — monitors per-dimension prediction accuracy; proposes weight adjustments.

Rolling 7-day accuracy vs 30-day baseline.
If drift > 0.05: POST /governance/weights/propose.
"""
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import httpx
import aiosqlite

from ..config import settings

logger = logging.getLogger(__name__)

_DIMENSIONS = ["policy_score", "outcome_score", "risk_score", "context_fit_score"]

_DEFAULT_WEIGHTS = {
    'policy_score': 0.4,
    'outcome_score': 0.3,
    'risk_score': 0.2,
    'context_fit_score': 0.1,
}


class ScoreAdapter:
    def __init__(self, db_path: Path) -> None:
        self._db = db_path

    async def evaluate(self) -> list[dict]:
        episodes = await self._fetch_episodes_with_scores()
        if not episodes:
            return []

        now = time.time()
        seven_days = now - 86400 * 7
        thirty_days = now - 86400 * 30

        short_win = [e for e in episodes if e["created_at"] >= seven_days]
        long_win = [e for e in episodes if e["created_at"] >= thirty_days]

        proposals = []
        for dim in _DIMENSIONS:
            short_acc = self._dimension_accuracy(short_win, dim)
            long_acc = self._dimension_accuracy(long_win, dim)
            drift = abs(short_acc - long_acc)

            logger.info(
                "Dimension %s: 7d=%.3f  30d=%.3f  drift=%.4f",
                dim, short_acc, long_acc, drift,
            )

            if drift > 0.05 and len(short_win) >= 10:
                logger.warning(
                    "Drift %.4f > 0.05 for %s — proposing weight adjustment",
                    drift, dim,
                )
                intent_classes = set(e["intent_class"] for e in short_win)
                for ic in intent_classes:
                    current = await self._fetch_current_weights(ic)
                    version = await self._propose_via_api(
                        intent_class=ic,
                        current_weights=current.get("weights", current),
                        dimension=dim,
                        accuracy=short_acc,
                    )
                    if version:
                        proposals.append({
                            "intent_class": ic,
                            "dimension": dim,
                            "drift": round(drift, 4),
                            "short_accuracy": round(short_acc, 4),
                            "long_accuracy": round(long_acc, 4),
                            "version": version,
                        })
        return proposals

    def _dimension_accuracy(self, episodes: list[dict], dimension: str) -> float:
        predicted = [ep["scores"].get(dimension) for ep in episodes
                     if ep.get("scores") and ep["scores"].get(dimension) is not None]
        if len(predicted) < 5:
            return 1.0
        actuals = [1.0 if ep["outcome"] == "SUCCESS" else 0.0 for ep in episodes
                   if ep.get("scores") and ep["scores"].get(dimension) is not None]
        return self._correlation(predicted, actuals)

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
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                 """SELECT episode_id, intent_class, outcome, context_snapshot, created_at
                    FROM decision_episodes
                   WHERE outcome IS NOT NULL
                     AND json_extract(context_snapshot, '$.scores') IS NOT NULL
                   ORDER BY created_at DESC
                   LIMIT 5000"""
            ) as cursor:
                rows = await cursor.fetchall()

        results = []
        for r in rows:
            try:
                snapshot = json.loads(r["context_snapshot"])
            except (json.JSONDecodeError, TypeError):
                continue
            scores = snapshot.get("scores") if isinstance(snapshot, dict) else None
            if not scores:
                continue
            results.append({
                "episode_id": r["episode_id"],
                "intent_class": r["intent_class"],
                "outcome": r["outcome"],
                "scores": scores,
                "created_at": r["created_at"],
            })
        return results

    async def _fetch_current_weights(self, intent_class: str) -> dict:
        async with aiosqlite.connect(self._db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT weights FROM weight_configs WHERE intent_class = ? AND is_active = 1",
                (intent_class,),
            ) as cursor:
                row = await cursor.fetchone()
        if row:
            return json.loads(row["weights"])
        return dict(_DEFAULT_WEIGHTS)

    async def _propose_via_api(
        self,
        intent_class: str,
        current_weights: dict,
        dimension: str,
        accuracy: float,
    ) -> str:
        rate = 0.6 - accuracy
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

        payload = {
            "intent_class": intent_class,
            "weights": new_weights,
            "episode_batch": f"batch-{int(time.time())}",
            "proposed_by": "score_adapter",
        }
        try:
            async with httpx.AsyncClient(base_url="http://localhost:8001", timeout=10.0) as client:
                resp = await client.post("/governance/weights/propose", json=payload)
                resp.raise_for_status()
                result = resp.json()
                logger.info("Weight proposal submitted: %s", result.get("version"))
                return result.get("version", "")
        except Exception as exc:
            logger.error("Failed to propose weight adjustment: %s", exc)
            return ""