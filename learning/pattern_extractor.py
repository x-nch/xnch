"""Pattern Extractor — 6h APScheduler job.

Groups completed episodes by (intent_class, action_type, entity_class, actor_role).
Requires MIN_OBSERVATIONS before writing a pattern.
Bayesian-smoothed confidence: (success_count + 1) / (observation_count + 2)
"""
import hashlib
import logging
from uuid import uuid4

from ..memory.episodic_store import EpisodicStore
from ..memory.pattern_store import PatternStore
from ..config import settings

logger = logging.getLogger(__name__)


def _context_signature(intent_class: str, action_type: str, entity_class: str, actor_role: str) -> str:
    canonical = "|".join([
        intent_class.lower(), action_type.lower(), entity_class.lower(), actor_role.lower()
    ])
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


class PatternExtractor:
    def __init__(self, episodic: EpisodicStore, patterns: PatternStore) -> None:
        self._episodic = episodic
        self._patterns = patterns

    async def run(self, tuples: list[tuple] | None = None) -> int:
        """Extract patterns for the given tuples (or all distinct tuples if None).
        Returns count of patterns written."""
        if tuples is None:
            tuples = await self._episodic.get_distinct_tuples()

        run_id = str(uuid4())
        written = 0

        for intent_class, action_type, entity_class, actor_role in tuples:
            episodes = await self._episodic.fetch_for_extraction(
                intent_class, action_type, entity_class, actor_role
            )
            n = len(episodes)
            if n < settings.pattern_min_observations:
                logger.debug(
                    "Skipping %s|%s|%s|%s — only %d episodes (need %d)",
                    intent_class, action_type, entity_class, actor_role,
                    n, settings.pattern_min_observations,
                )
                continue

            successes = sum(1 for ep in episodes if ep["outcome"] == "SUCCESS")
            success_rate = successes / n
            confidence = (successes + 1) / (n + 2)  # Bayesian smoothing

            deltas = [ep["prediction_delta"] for ep in episodes if ep["prediction_delta"] is not None]
            avg_delta = sum(deltas) / len(deltas) if deltas else None

            sig = _context_signature(intent_class, action_type, entity_class, actor_role)

            await self._patterns.upsert_pattern(
                context_signature=sig,
                intent_class=intent_class,
                action_type=action_type,
                entity_class=entity_class,
                actor_role=actor_role,
                success_rate=round(success_rate, 4),
                confidence=round(confidence, 4),
                observation_count=n,
                avg_prediction_delta=round(avg_delta, 4) if avg_delta is not None else None,
                extraction_run_id=run_id,
            )
            written += 1
            logger.info(
                "Pattern written: %s|%s|%s|%s  success_rate=%.2f  confidence=%.2f  n=%d",
                intent_class, action_type, entity_class, actor_role,
                success_rate, confidence, n,
            )

        return written

    async def run_early(self) -> int:
        """Run for all tuples flagged for early extraction, then clear flags."""
        flagged = await self._episodic.get_flagged_for_early_extraction()
        if not flagged:
            return 0
        written = await self.run(tuples=flagged)
        await self._episodic.clear_early_extraction_flags()
        return written
