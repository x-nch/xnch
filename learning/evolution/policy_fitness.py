"""Policy rule fitness — outcome consistency.

A rule's conditions select a subset of decision episodes. Fitness measures how
often the observed outcome aligns with the rule's verdict:
- ALLOW / ALLOW_WITH_WARNINGS / MODIFY favor episodes that succeed.
- BLOCK / DEFER favor episodes that fail.

Neutral 0.5 when no episodes match or observations are too few.
"""
from .policy_rule import PolicyRule

_MIN_OBSERVATIONS = 5

_POSITIVE_VERDICTS = {"ALLOW", "ALLOW_WITH_WARNINGS", "MODIFY"}
_NEGATIVE_VERDICTS = {"BLOCK", "DEFER"}


def _outcome_score(outcome: str) -> float:
    if outcome == "SUCCESS":
        return 1.0
    if outcome == "FAILURE":
        return 0.0
    return 0.5  # PARTIAL / unknown → neutral


def compute_rule_fitness(rule: PolicyRule, episodes: list[dict]) -> float:
    matching = [ep for ep in episodes if rule.matches(ep)]
    if len(matching) < _MIN_OBSERVATIONS:
        return 0.5

    success_rate = sum(
        1 for ep in matching if ep.get("outcome") == "SUCCESS"
    ) / len(matching)

    if rule.verdict in _NEGATIVE_VERDICTS:
        # Failures are the desired outcome for blocking/deferring rules.
        return 1.0 - success_rate
    return success_rate
