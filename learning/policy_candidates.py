"""Policy Candidate Generator — produces soft policy candidates for operator review."""
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import aiosqlite

from ..memory.pattern_store import PatternStore

logger = logging.getLogger(__name__)


class PolicyCandidateGenerator:
    def __init__(self, patterns: PatternStore, db_path: Path) -> None:
        self._patterns = patterns
        self._db = db_path

    async def run(self) -> int:
        """Generate candidates for low-success, high-confidence patterns."""
        low_success = await self._patterns.fetch_low_success(
            max_success_rate=0.4, min_confidence=0.6
        )
        count = 0
        for pattern in low_success:
            if await self._candidate_exists(pattern["pattern_id"]):
                continue

            rule_yaml = self._generate_candidate_rule(pattern)
            candidate_id = str(uuid4())

            async with aiosqlite.connect(self._db) as db:
                await db.execute(
                    """INSERT INTO policy_candidates
                       (candidate_id, pattern_id, rule_yaml, triggering_pattern, status, created_at)
                       VALUES (?, ?, ?, ?, 'PENDING', ?)""",
                    (candidate_id, pattern["pattern_id"], rule_yaml,
                     json.dumps(pattern), time.time()),
                )
                await db.commit()

            logger.info(
                "Policy candidate generated for pattern %s (success_rate=%.2f, confidence=%.2f)",
                pattern["pattern_id"], pattern["success_rate"], pattern["confidence"],
            )
            count += 1
        return count

    def _generate_candidate_rule(self, pattern: dict) -> str:
        return (
            f"# Auto-generated candidate — requires operator review\n"
            f"# Pattern: {pattern['intent_class']}|{pattern['action_type']}|"
            f"{pattern['entity_class']}|{pattern['actor_role']}\n"
            f"# success_rate={pattern['success_rate']:.2f}  "
            f"confidence={pattern['confidence']:.2f}  n={pattern['observation_count']}\n"
            f"- rule_id: candidate-{pattern['pattern_id'][:8]}\n"
            f"  priority: 100\n"
            f"  conditions:\n"
            f"    intent_class: {pattern['intent_class']}\n"
            f"    action_type: {pattern['action_type']}\n"
            f"    entity_class: {pattern['entity_class']}\n"
            f"    actor_role: {pattern['actor_role']}\n"
            f"  action:\n"
            f"    verdict: ALLOW_WITH_WARNINGS\n"
            f"    reason: Low historical success rate ({pattern['success_rate']:.0%}) — review recommended\n"
            f"    warnings:\n"
            f"      - Historical success rate {pattern['success_rate']:.0%} "
            f"(n={pattern['observation_count']})\n"
        )

    async def _candidate_exists(self, pattern_id: str) -> bool:
        async with aiosqlite.connect(self._db) as db:
            async with db.execute(
                "SELECT 1 FROM policy_candidates WHERE pattern_id = ? AND status = 'PENDING'",
                (pattern_id,),
            ) as cursor:
                return await cursor.fetchone() is not None
