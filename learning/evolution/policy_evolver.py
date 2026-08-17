"""PolicyRuleEvolver — evolutionary search over policy DSL rules.

Evolves a population of PolicyRule candidates over decision episodes using
Boltzmann selection (reusing boltzmann.py) and MAP-Elites archive (keyed by
intent_class cell). Winners that beat a fitness threshold are written to the
policy_candidates table for operator review.
"""
import logging
from uuid import uuid4

from .boltzmann import annealing_temperature, select_parent_index
from .map_elites import MapElitesArchive
from .policy_fitness import compute_rule_fitness
from .policy_rule import PolicyRule, random_policy_rule

logger = logging.getLogger(__name__)

_INITIAL_TEMPERATURE = 1.0
_TEMPERATURE_DECAY = 0.9
_DEFAULT_POPULATION = 30
_DEFAULT_GENERATIONS = 10
_DEFAULT_MIN_FITNESS = 0.75
_DEFAULT_BATCH = 5


def _rule_cell(rule: PolicyRule) -> tuple:
    """MAP-Elites cell key: intent_class condition (or '*' for wildcard)."""
    return (rule.conditions.get("intent_class", "*"), rule.verdict)


class PolicyRuleEvolver:
    def __init__(
        self,
        episodes_fn=None,
        candidate_write_fn=None,
        population_size: int = _DEFAULT_POPULATION,
        generations: int = _DEFAULT_GENERATIONS,
        min_fitness: float = _DEFAULT_MIN_FITNESS,
        batch_size: int = _DEFAULT_BATCH,
        rng=None,
    ) -> None:
        self._episodes = episodes_fn or self._default_fetch_episodes
        self._write_candidate = candidate_write_fn or self._default_write_candidate
        self._population_size = population_size
        self._generations = generations
        self._min_fitness = min_fitness
        self._batch_size = batch_size
        self._rng = rng

    async def run(self) -> list[dict]:
        episodes = await self._episodes()
        if len(episodes) < 5:
            logger.info("PolicyEvolver: only %d episodes — skipping", len(episodes))
            return []

        population = [random_policy_rule(self._rng) for _ in range(self._population_size)]
        fitnesses = [compute_rule_fitness(rule, episodes) for rule in population]

        archive = MapElitesArchive(descriptor_fn=_rule_cell)
        for rule, fit in zip(population, fitnesses):
            archive.add(rule, fit)

        for gen in range(self._generations):
            temperature = annealing_temperature(_INITIAL_TEMPERATURE, gen, _TEMPERATURE_DECAY)
            parent_idx = select_parent_index(fitnesses, temperature, self._rng)
            child = population[parent_idx].mutate(self._rng)
            child_fitness = compute_rule_fitness(child, episodes)
            archive.add(child, child_fitness)

            min_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if child_fitness > fitnesses[min_idx]:
                population[min_idx] = child
                fitnesses[min_idx] = child_fitness

        proposals = []
        for rule, fit in archive.all_individuals():
            if fit < self._min_fitness:
                continue
            rule_yaml = rule.to_yaml()
            candidate_id = await self._write_candidate(
                candidate_id=str(uuid4()),
                rule_yaml=rule_yaml,
                triggering_pattern={
                    "intent_class": rule.conditions.get("intent_class", "*"),
                    "action_type": rule.conditions.get("action_type", "*"),
                    "entity_class": rule.conditions.get("entity_class", "*"),
                    "actor_role": rule.conditions.get("actor_role", "*"),
                    "fitness": round(fit, 4),
                },
            )
            logger.info("PolicyEvolver: proposed %s fitness=%.3f → %s",
                        rule_yaml.strip(), fit, candidate_id)
            proposals.append({
                "candidate_id": candidate_id,
                "rule_yaml": rule_yaml,
                "fitness": round(fit, 4),
            })
            if len(proposals) >= self._batch_size:
                break

        return proposals

    # ------------------------------------------------------------------ #
    # Production defaults
    # ------------------------------------------------------------------ #

    async def _default_fetch_episodes(self) -> list[dict]:
        from datetime import datetime, timedelta, timezone

        from ..memory.pg_episodic_store import PgEpisodicStore

        store = PgEpisodicStore()
        await store.connect()
        try:
            since = datetime.now(timezone.utc) - timedelta(days=30)
            return await store.fetch_decision_episodes_with_scores(since)
        finally:
            await store.close()

    async def _default_write_candidate(
        self,
        candidate_id: str,
        rule_yaml: str,
        triggering_pattern: dict,
    ) -> str:
        import json
        import time

        import aiosqlite

        from ..config import settings

        async with aiosqlite.connect(settings.base_dir / "xnch.db") as db:
            await db.execute(
                """INSERT INTO policy_candidates
                   (candidate_id, pattern_id, rule_yaml, triggering_pattern, status, created_at)
                   VALUES (?, ?, ?, ?, 'PENDING', ?)""",
                (candidate_id, "evolved", rule_yaml, json.dumps(triggering_pattern), time.time()),
            )
            await db.commit()
        return candidate_id
