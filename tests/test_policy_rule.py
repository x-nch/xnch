"""Policy rule genotype + mutation tests."""
import pytest

from xnch.learning.evolution.policy_rule import (
    PolicyRule,
    random_policy_rule,
    VALID_VERDICTS,
)

_CONDITIONS = {
    "intent_class": "EXECUTION",
    "action_type": "DEPLOY",
    "entity_class": "SERVICE",
    "actor_role": "operator",
}


def test_rule_requires_valid_verdict():
    with pytest.raises(ValueError):
        PolicyRule(conditions=_CONDITIONS, verdict="DENY")
    with pytest.raises(ValueError):
        PolicyRule(conditions=_CONDITIONS, verdict="")


def test_rule_accepts_wildcard_conditions():
    rule = PolicyRule(conditions={}, verdict="BLOCK")
    assert rule.conditions == {}


def test_rule_matches_exact_episode():
    rule = PolicyRule(conditions=_CONDITIONS, verdict="BLOCK")
    episode = {"intent_class": "EXECUTION", "action_type": "DEPLOY",
               "entity_class": "SERVICE", "actor_role": "operator"}
    assert rule.matches(episode)


def test_rule_does_not_match_wrong_episode():
    rule = PolicyRule(conditions=_CONDITIONS, verdict="BLOCK")
    episode = {"intent_class": "QUERY", "action_type": "LIST",
               "entity_class": "FILE", "actor_role": "viewer"}
    assert not rule.matches(episode)


def test_rule_wildcard_matches_any():
    rule = PolicyRule(conditions={"intent_class": "EXECUTION"}, verdict="ALLOW")
    assert rule.matches({"intent_class": "EXECUTION", "action_type": "DEPLOY",
                         "entity_class": "SERVICE", "actor_role": "operator"})
    assert rule.matches({"intent_class": "EXECUTION", "action_type": "ROLLBACK",
                         "entity_class": "ML_MODEL", "actor_role": "admin"})


def test_rule_to_yaml_roundtrip_fields():
    rule = PolicyRule(conditions=_CONDITIONS, verdict="MODIFY",
                      reason="Require backup first")
    y = rule.to_yaml()
    assert "MODIFY" in y
    assert "DEPLOY" in y
    assert "Require backup first" in y


def test_mutation_keeps_valid_verdict_and_shape():
    rule = PolicyRule(conditions=_CONDITIONS, verdict="ALLOW")
    for _ in range(30):
        child = rule.mutate()
        assert child.verdict in VALID_VERDICTS
        assert set(child.conditions.keys()).issubset(
            {"intent_class", "action_type", "entity_class", "actor_role"}
        )


def test_mutation_sometimes_changes_verdict():
    rule = PolicyRule(conditions=_CONDITIONS, verdict="ALLOW")
    changed = sum(1 for _ in range(100) if rule.mutate().verdict != "ALLOW")
    assert changed > 0


def test_random_rule_is_valid():
    for _ in range(20):
        rule = random_policy_rule()
        assert rule.verdict in VALID_VERDICTS
        assert rule.priority > 0
