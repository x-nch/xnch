"""Policy Candidate Generator — LLM-driven policy rule proposals.

Reads low-success, high-confidence patterns from PatternStore.
Calls LLM (via litellm proxy) to generate policy DSL candidate rules.
Outputs candidates with ALLOW/BLOCK/MODIFY/DEFER verdicts (never DENY).
Writes to policy_candidates SQLite table.
"""
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import httpx
import aiosqlite
import yaml

from ..memory.pattern_store import PatternStore
from ..config import settings

logger = logging.getLogger(__name__)

_LLM_URL = settings.litellm_proxy_url.rstrip("/") + "/chat/completions"

_LLM_MODEL = "ornith"
_CANDIDATE_SYSTEM_PROMPT = """You are a policy engineer for the XNCH autonomous governance system.
Given failure patterns, suggest policy rule candidates in XNCH policy DSL.

Rules have this YAML structure:
```yaml
- rule_id: <unique-id>
  priority: <integer — 100-500, lower = higher priority>
  conditions:
    intent_class: <EXECUTION | QUERY | DECISION | ESCALATION>
    action_type: <string or omit>
    entity_class: <string or omit>
    actor_role: <string or omit>
  action:
    verdict: <ALLOW | ALLOW_WITH_WARNINGS | BLOCK | MODIFY | DEFER>
    reason: <string>
```

Constraints:
- Use ONLY these verdicts: ALLOW, ALLOW_WITH_WARNINGS, BLOCK, MODIFY, DEFER (never DENY)
- ALLOW_WITH_WARNINGS: when action is ok but caution needed
- BLOCK: when pattern suggests action should be prevented
- MODIFY: when action spec needs field changes (include modify_spec)
- DEFER: when action needs human approval (include requires_actor)
- Generate 1-3 candidate rules per batch
- Output raw YAML only, no markdown fences
"""


class PolicyCandidateGenerator:
    def __init__(self, patterns: PatternStore, db_path: Path) -> None:
        self._patterns = patterns
        self._db = db_path

    async def run(self) -> int:
        low_success = await self._patterns.fetch_low_success(
            max_success_rate=0.4, min_confidence=0.6
        )
        if not low_success:
            logger.info("No low-success patterns found")
            return 0

        count = 0
        for i in range(0, len(low_success), 5):
            batch = low_success[i:i + 5]
            if all(await self._candidate_exists(p["pattern_id"]) for p in batch):
                continue

            yaml_output = await self._llm_generate(batch)
            if not yaml_output:
                continue

            candidates = self._parse_candidates(yaml_output, batch)
            for cand in candidates:
                async with aiosqlite.connect(self._db) as db:
                    await db.execute(
                        """INSERT INTO policy_candidates
                           (candidate_id, pattern_id, rule_yaml, triggering_pattern, status, created_at)
                           VALUES (?, ?, ?, ?, 'PENDING', ?)""",
                        (cand["candidate_id"], cand["pattern_id"], cand["rule_yaml"],
                         json.dumps(cand["triggering_pattern"]), time.time()),
                    )
                    await db.commit()
                logger.info("Policy candidate %s written", cand["candidate_id"])
                count += 1
        return count

    async def _llm_generate(self, patterns: list[dict]) -> str | None:
        pattern_summary = "\n".join(
            f"- {p['intent_class']}|{p['action_type']}|{p['entity_class']}|{p['actor_role']}  "
            f"success_rate={p['success_rate']:.2f}  confidence={p['confidence']:.2f}  "
            f"n={p['observation_count']}"
            for p in patterns
        )
        user_msg = f"Generate policy rules for these failure patterns:\n{pattern_summary}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(_LLM_URL, json={
                    "model": _LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": _CANDIDATE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1500,
                })
                resp.raise_for_status()
                body = resp.json()
                content = body["choices"][0]["message"]["content"]
                return content.strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return None

    def _parse_candidates(self, yaml_text: str, patterns: list[dict]) -> list[dict]:
        if yaml_text.startswith("```"):
            yaml_text = yaml_text.strip().removeprefix("```yaml").removeprefix("```")
            if "```" in yaml_text:
                yaml_text = yaml_text[:yaml_text.index("```")].strip()

        try:
            raw = yaml.safe_load(yaml_text)
        except Exception as exc:
            logger.warning("Failed to parse LLM YAML output: %s", exc)
            logger.debug("Raw output: %s", yaml_text)
            return []

        if not raw:
            return []

        rules = raw if isinstance(raw, list) else raw.get("rules", [])
        if not isinstance(rules, list):
            rules = [rules]

        candidates = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            verdict = rule.get("action", {}).get("verdict", "")
            if verdict == "DENY":
                continue

            pattern_id = self._match_pattern_id(rule, patterns)
            cand_id = str(uuid4())
            rule_yaml = yaml.dump([rule], default_flow_style=False)

            candidates.append({
                "candidate_id": cand_id,
                "pattern_id": pattern_id,
                "rule_yaml": rule_yaml,
                "triggering_pattern": next(
                    (p for p in patterns if p["pattern_id"] == pattern_id), patterns[0]
                ),
            })
        return candidates

    def _match_pattern_id(self, rule: dict, patterns: list[dict]) -> str:
        cond = rule.get("conditions", {})
        ic = cond.get("intent_class", "")
        at = cond.get("action_type", "")
        ec = cond.get("entity_class", "")
        ar = cond.get("actor_role", "")
        for p in patterns:
            if (p["intent_class"] == ic and p["action_type"] == at
                    and p["entity_class"] == ec and p["actor_role"] == ar):
                return p["pattern_id"]
        return patterns[0]["pattern_id"] if patterns else ""

    async def _candidate_exists(self, pattern_id: str) -> bool:
        async with aiosqlite.connect(self._db) as db:
            async with db.execute(
                "SELECT 1 FROM policy_candidates WHERE pattern_id = ? AND status = 'PENDING'",
                (pattern_id,),
            ) as cursor:
                return await cursor.fetchone() is not None