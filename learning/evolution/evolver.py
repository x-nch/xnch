"""WeightEvolver — orchestrator for evolutionary search over decision weights.

Replaces the single-dimension drift heuristic of score_adapter. Groups scored
episodes by intent class, runs island evolution per class (with MAP-Elites
archive + Boltzmann selection), and proposes a winner via the governance
weights endpoint when it beats the currently active config by a margin.
"""
import logging
import time
from typing import Any

from .fitness import compute_fitness
from .islands import evolve_island

logger = logging.getLogger(__name__)

_INTENT_CLASSES = ["EXECUTION", "QUERY", "DECISION", "ESCALATION"]
_MIN_EPISODES = 10
_MIN_FITNESS_GAIN = 0.05


class WeightEvolver:
    def __init__(
        self,
        fetch_fn=None,
        propose_fn=None,
        current_weights_fn=None,
        population_size: int = 20,
        generations: int = 10,
        min_fitness_gain: float = _MIN_FITNESS_GAIN,
    ) -> None:
        self._fetch = fetch_fn or self._default_fetch
        self._propose = propose_fn or self._default_propose
        self._current = current_weights_fn or self._default_current_weights
        self._population_size = population_size
        self._generations = generations
        self._min_fitness_gain = min_fitness_gain

    async def run(self) -> list[dict[str, Any]]:
        episodes = await self._fetch()
        results = []

        for intent_class in _INTENT_CLASSES:
            class_episodes = [e for e in episodes if e.get("intent_class") == intent_class]
            if len(class_episodes) < _MIN_EPISODES:
                logger.info("Evolver: %s has %d episodes (min %d) — skipping",
                            intent_class, len(class_episodes), _MIN_EPISODES)
                continue

            async def fitness_fn(ind, class_episodes=class_episodes):
                return compute_fitness(ind.weights, class_episodes)

            current = await self._current(intent_class)
            current_weights = current.get("weights", {}) if current else {}
            current_fitness = compute_fitness(current_weights, class_episodes)

            winner = await evolve_island(
                intent_class=intent_class,
                fitness_fn=fitness_fn,
                population_size=self._population_size,
                generations=self._generations,
            )
            if winner is None:
                continue

            best_ind, best_fitness = winner
            if best_fitness - current_fitness < self._min_fitness_gain:
                logger.info(
                    "Evolver: %s winner fitness %.4f not better than current %.4f — skipping",
                    intent_class, best_fitness, current_fitness,
                )
                continue

            version = await self._propose(
                intent_class=intent_class,
                weights=best_ind.weights,
                episode_batch=f"evolve-{int(time.time())}",
            )
            if not version:
                continue

            logger.info(
                "Evolver: proposed %s weights %s (fitness %.4f vs current %.4f) → %s",
                intent_class, best_ind.weights, best_fitness, current_fitness, version,
            )
            results.append({
                "intent_class": intent_class,
                "version": version,
                "winner_fitness": round(best_fitness, 4),
                "current_fitness": round(current_fitness, 4),
                "weights": best_ind.weights,
            })

        return results

    # ------------------------------------------------------------------ #
    # Production defaults (overridable in tests)
    # ------------------------------------------------------------------ #

    async def _default_fetch(self) -> list[dict[str, Any]]:
        from datetime import datetime, timedelta, timezone

        from ..memory.pg_episodic_store import PgEpisodicStore

        store = PgEpisodicStore()
        await store.connect()
        try:
            since = datetime.now(timezone.utc) - timedelta(days=30)
            return await store.fetch_decision_episodes_with_scores(since)
        finally:
            await store.close()

    async def _default_current_weights(self, intent_class: str) -> dict[str, Any]:
        import aiosqlite

        from ..config import settings

        async with aiosqlite.connect(settings.base_dir / "xnch.db") as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT version, weights FROM weight_configs "
                "WHERE intent_class = ? AND is_active = 1",
                (intent_class,),
            ) as cursor:
                row = await cursor.fetchone()
        if not row:
            return {}
        return {"version": row["version"], "weights": _parse_weights(row["weights"])}

    async def _default_propose(self, intent_class: str, weights: dict, episode_batch: str) -> str:
        import httpx

        from ..config import settings

        payload = {
            "intent_class": intent_class,
            "weights": weights,
            "episode_batch": episode_batch,
            "proposed_by": "weight_evolver",
        }
        try:
            async with httpx.AsyncClient(
                base_url=settings.self_base_url, timeout=10.0
            ) as client:
                resp = await client.post("/governance/weights/propose", json=payload)
                resp.raise_for_status()
                return resp.json().get("version", "")
        except Exception as exc:
            logger.error("Failed to propose weight config (%s): %s", intent_class, exc)
            return ""


def _parse_weights(raw: str) -> dict[str, float]:
    import json

    try:
        data = json.loads(raw)
        return {k: float(v) for k, v in data.items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
