"""Contract 1: YAML policy loader. Merges default + custom rule sets."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TimeWindow:
    days: list[str] = field(default_factory=list)
    hours_utc: str = ""


@dataclass
class RuleConditions:
    intent_class: str | None = None
    action_type: str | None = None
    entity_class: str | None = None
    actor_role: str | None = None
    actor_capabilities: list[str] = field(default_factory=list)
    urgency: str | None = None
    reversible: bool | None = None
    time_window: TimeWindow | None = None


@dataclass
class RuleAction:
    verdict: str
    reason: str
    warnings: list[str] = field(default_factory=list)
    modify_spec: dict[str, Any] | None = None
    requires_actor: str | None = None


@dataclass
class Rule:
    rule_id: str
    priority: int
    conditions: RuleConditions
    action: RuleAction
    source_file: str = ""


@dataclass
class PolicySet:
    rules: list[Rule] = field(default_factory=list)


def _parse_rule(raw: dict, source: str) -> Rule:
    cond_raw = raw.get("conditions") or {}
    tw_raw = cond_raw.get("time_window")
    conditions = RuleConditions(
        intent_class=cond_raw.get("intent_class"),
        action_type=cond_raw.get("action_type"),
        entity_class=cond_raw.get("entity_class"),
        actor_role=cond_raw.get("actor_role"),
        actor_capabilities=cond_raw.get("actor_capabilities") or [],
        urgency=cond_raw.get("urgency"),
        reversible=cond_raw.get("reversible"),
        time_window=TimeWindow(
            days=tw_raw.get("days", []),
            hours_utc=tw_raw.get("hours_utc", ""),
        ) if tw_raw else None,
    )
    act_raw = raw.get("action", {})
    action = RuleAction(
        verdict=act_raw["verdict"],
        reason=act_raw.get("reason", ""),
        warnings=act_raw.get("warnings") or [],
        modify_spec=act_raw.get("modify_spec"),
        requires_actor=act_raw.get("requires_actor"),
    )
    return Rule(
        rule_id=raw["rule_id"],
        priority=raw["priority"],
        conditions=conditions,
        action=action,
        source_file=source,
    )


def _load_file(path: Path) -> list[Rule]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text())
    rules = []
    for raw in (data or {}).get("rules", []):
        try:
            rules.append(_parse_rule(raw, path.name))
        except Exception as exc:
            logger.warning("Skipping malformed rule in %s: %s", path.name, exc)
    return rules


class PolicyLoader:
    def __init__(self, policies_dir: Path) -> None:
        self._dir = policies_dir

    def load(self) -> PolicySet:
        default_rules = _load_file(self._dir / "default.yaml")
        custom_rules = _load_file(self._dir / "custom.yaml")
        all_rules = default_rules + custom_rules

        # Warn on duplicate priorities (per Contract 1)
        seen: dict[int, str] = {}
        for r in all_rules:
            if r.priority in seen:
                logger.warning(
                    "Duplicate priority %d: rule '%s' and '%s'",
                    r.priority, seen[r.priority], r.rule_id,
                )
            else:
                seen[r.priority] = r.rule_id

        return PolicySet(rules=sorted(all_rules, key=lambda r: r.priority))
