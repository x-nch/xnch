"""Contract 1: policy evaluation engine.

Rules evaluated in ascending priority order. First match wins.
Default ALLOW when no rule matches.
"""
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .loader import PolicySet, Rule, RuleConditions


_DAY_MAP = {
    "MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6,
}


class PolicyVerdict:
    ALLOW = "ALLOW"
    ALLOW_WITH_WARNINGS = "ALLOW_WITH_WARNINGS"
    MODIFY = "MODIFY"
    DEFER = "DEFER"
    BLOCK = "BLOCK"


@dataclass
class DryRunResult:
    verdict: str
    policy_refs: list[str]
    warnings: list[str] = field(default_factory=list)
    modified_action_spec: dict[str, Any] | None = None
    requires_actor: str | None = None


def _matches_time_window(tw) -> bool:
    if not tw:
        return True
    now = time.gmtime()
    if tw.days:
        if time.strftime("%a", now).upper()[:3] not in tw.days:
            return False
    if tw.hours_utc:
        m = re.match(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", tw.hours_utc)
        if m:
            start_min = int(m.group(1)) * 60 + int(m.group(2))
            end_min = int(m.group(3)) * 60 + int(m.group(4))
            now_min = now.tm_hour * 60 + now.tm_min
            if not (start_min <= now_min < end_min):
                return False
    return True


def _matches(rule: Rule, ctx: dict[str, Any]) -> bool:
    c = rule.conditions
    if c.intent_class and c.intent_class != ctx.get("intent_class"):
        return False
    if c.action_type and c.action_type.lower() != (ctx.get("action_type") or "").lower():
        return False
    if c.entity_class and c.entity_class != ctx.get("entity_class"):
        return False
    if c.actor_role and c.actor_role != ctx.get("actor_role"):
        return False
    if c.actor_capabilities:
        caps = set(ctx.get("actor_capabilities") or [])
        if not all(cap in caps for cap in c.actor_capabilities):
            return False
    if c.urgency and c.urgency != ctx.get("urgency"):
        return False
    if c.reversible is not None and c.reversible != ctx.get("reversible"):
        return False
    if not _matches_time_window(c.time_window):
        return False
    return True


def _apply_modify(action_spec: dict[str, Any], modify_spec: dict[str, Any]) -> dict[str, Any]:
    """Apply dot-path field replacement to a copy of action_spec."""
    result = dict(action_spec)
    dot_path: str = modify_spec.get("field", "")
    value = modify_spec.get("value")
    parts = dot_path.split(".")
    target = result
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    if parts:
        target[parts[-1]] = value
    return result


class PolicyEngine:
    def __init__(self, policy_set: PolicySet) -> None:
        self._rules = policy_set.rules

    def evaluate(
        self,
        intent_class: str,
        action_type: str,
        entity_class: str,
        actor_role: str,
        actor_capabilities: list[str],
        action_spec: dict[str, Any],
        urgency: str = "NORMAL",
        reversible: bool = True,
    ) -> DryRunResult:
        ctx = {
            "intent_class": intent_class,
            "action_type": action_type,
            "entity_class": entity_class,
            "actor_role": actor_role,
            "actor_capabilities": actor_capabilities,
            "urgency": urgency,
            "reversible": reversible,
        }

        for rule in self._rules:
            if not _matches(rule, ctx):
                continue

            verdict = rule.action.verdict

            if verdict == PolicyVerdict.BLOCK:
                return DryRunResult(verdict=verdict, policy_refs=[rule.rule_id])

            if verdict == PolicyVerdict.MODIFY:
                modified = _apply_modify(action_spec, rule.action.modify_spec or {})
                return DryRunResult(
                    verdict=verdict,
                    policy_refs=[rule.rule_id],
                    modified_action_spec=modified,
                )

            if verdict == PolicyVerdict.DEFER:
                return DryRunResult(
                    verdict=verdict,
                    policy_refs=[rule.rule_id],
                    requires_actor=rule.action.requires_actor,
                )

            if verdict == PolicyVerdict.ALLOW_WITH_WARNINGS:
                return DryRunResult(
                    verdict=verdict,
                    policy_refs=[rule.rule_id],
                    warnings=rule.action.warnings,
                )

            # ALLOW
            return DryRunResult(verdict=PolicyVerdict.ALLOW, policy_refs=[rule.rule_id])

        # Default: no rule matched
        return DryRunResult(verdict=PolicyVerdict.ALLOW, policy_refs=["default.allow"])

    def reload(self, policy_set: PolicySet) -> None:
        self._rules = policy_set.rules
