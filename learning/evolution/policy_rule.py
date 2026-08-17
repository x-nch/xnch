"""Policy rule genotype for evolutionary search over policy DSL rules.

A rule is a conditions tuple (each field optional/wildcard) + verdict. Fitness
is evaluated over decision episodes: how often episodes matching the rule's
conditions had outcomes consistent with the rule's verdict.
"""
import random
import uuid
import yaml

VALID_VERDICTS = ["ALLOW", "ALLOW_WITH_WARNINGS", "BLOCK", "MODIFY", "DEFER"]
_CONDITION_KEYS = ["intent_class", "action_type", "entity_class", "actor_role"]
_INTENT_CLASSES = ["EXECUTION", "QUERY", "DECISION", "ESCALATION"]
_ACTION_TYPES = ["DEPLOY", "ROLLBACK", "STAGE", "BACKUP", "RESTORE",
                 "READ_FILE", "WRITE_FILE", "LIST", "MUTATE", "RUN_SCRIPT"]
_ENTITY_CLASSES = ["SERVICE", "ML_MODEL", "DATABASE", "SCHEMA", "FILE", "CLUSTER"]
_ROLES = ["operator", "agent", "admin", "viewer", "nexi"]

_MUTATION_RATE = 0.3


class PolicyRule:
    def __init__(
        self,
        conditions: dict[str, str],
        verdict: str,
        priority: int = 100,
        reason: str = "",
    ) -> None:
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {VALID_VERDICTS}, got {verdict!r}")
        self.conditions = dict(conditions)
        self.verdict = verdict
        self.priority = priority
        self.reason = reason

    def matches(self, episode: dict) -> bool:
        for key in _CONDITION_KEYS:
            want = self.conditions.get(key)
            if want and want != episode.get(key):
                return False
        return True

    def to_yaml(self) -> str:
        rule = {
            "rule_id": f"evolved-{uuid.uuid4().hex[:8]}",
            "priority": self.priority,
            "conditions": dict(self.conditions),
            "action": {
                "verdict": self.verdict,
                "reason": self.reason or "Evolved policy candidate",
            },
        }
        return yaml.dump([rule], default_flow_style=False)

    def mutate(self, rng: random.Random | None = None) -> "PolicyRule":
        rng = rng or random.Random()
        verdict = self.verdict
        conditions = dict(self.conditions)

        if rng.random() < _MUTATION_RATE:
            verdict = rng.choice([v for v in VALID_VERDICTS if v != verdict])

        if rng.random() < _MUTATION_RATE:
            # Toggle a condition: add one if any missing, else drop a random one.
            missing = [k for k in _CONDITION_KEYS if k not in conditions]
            if missing and (not conditions or rng.random() < 0.5):
                key = rng.choice(missing)
                conditions[key] = rng.choice(_values_for(key))
            else:
                key = rng.choice(list(conditions.keys()))
                del conditions[key]

        return PolicyRule(
            conditions=conditions,
            verdict=verdict,
            priority=self.priority,
            reason=self.reason,
        )

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, PolicyRule)
                and self.conditions == other.conditions
                and self.verdict == other.verdict)


def _values_for(key: str) -> list[str]:
    if key == "intent_class":
        return _INTENT_CLASSES
    if key == "action_type":
        return _ACTION_TYPES
    if key == "entity_class":
        return _ENTITY_CLASSES
    return _ROLES


def random_policy_rule(rng: random.Random | None = None) -> PolicyRule:
    rng = rng or random.Random()
    conditions = {
        key: rng.choice(_values_for(key))
        for key in _CONDITION_KEYS
        if rng.random() < 0.5
    }
    return PolicyRule(
        conditions=conditions,
        verdict=rng.choice(VALID_VERDICTS),
        priority=rng.randint(100, 500),
    )
