"""Contract 1 — Policy engine tests."""
import pytest

from xnch.policy.loader import PolicyLoader, PolicySet, Rule, RuleConditions, RuleAction
from xnch.policy.engine import PolicyEngine, PolicyVerdict


def _make_engine(rules: list[Rule]) -> PolicyEngine:
    return PolicyEngine(PolicySet(rules=sorted(rules, key=lambda r: r.priority)))


def _rule(rule_id: str, priority: int, verdict: str, **cond_kwargs) -> Rule:
    conditions = RuleConditions(**cond_kwargs)
    action = RuleAction(verdict=verdict, reason="test")
    return Rule(rule_id=rule_id, priority=priority, conditions=conditions, action=action)


def test_default_allow_when_no_rules():
    engine = _make_engine([])
    result = engine.evaluate("EXECUTION", "DEPLOY", "SERVICE", "OPERATOR", [], {})
    assert result.verdict == PolicyVerdict.ALLOW


def test_exact_match_block():
    rules = [_rule("block-deploy", 10, "BLOCK", action_type="DEPLOY")]
    engine = _make_engine(rules)
    result = engine.evaluate("EXECUTION", "DEPLOY", "SERVICE", "OPERATOR", [], {})
    assert result.verdict == PolicyVerdict.BLOCK
    assert "block-deploy" in result.policy_refs


def test_priority_order_first_wins():
    rules = [
        _rule("allow-first", 5, "ALLOW", actor_role="OPERATOR"),
        _rule("block-second", 10, "BLOCK", actor_role="OPERATOR"),
    ]
    engine = _make_engine(rules)
    result = engine.evaluate("EXECUTION", "DEPLOY", "SERVICE", "OPERATOR", [], {})
    assert result.verdict == PolicyVerdict.ALLOW


def test_modify_replaces_action_spec():
    from xnch.policy.loader import RuleAction
    action = RuleAction(
        verdict="MODIFY",
        reason="force resource check",
        modify_spec={"field": "resource_check", "value": True},
    )
    rule = Rule(rule_id="force-check", priority=10,
                conditions=RuleConditions(action_type="DEPLOY"),
                action=action)
    engine = _make_engine([rule])
    result = engine.evaluate("EXECUTION", "DEPLOY", "SERVICE", "OPERATOR", [],
                             {"resource_check": False})
    assert result.verdict == PolicyVerdict.MODIFY
    assert result.modified_action_spec["resource_check"] is True


def test_allow_with_warnings_returns_warnings():
    action = RuleAction(
        verdict="ALLOW_WITH_WARNINGS",
        reason="risky",
        warnings=["This is risky"],
    )
    rule = Rule(rule_id="warn-rule", priority=10,
                conditions=RuleConditions(reversible=False),
                action=action)
    engine = _make_engine([rule])
    result = engine.evaluate("EXECUTION", "DEPLOY", "SERVICE", "OPERATOR", [], {},
                             reversible=False)
    assert result.verdict == PolicyVerdict.ALLOW_WITH_WARNINGS
    assert "This is risky" in result.warnings


def test_capabilities_all_required():
    rules = [
        _rule("require-caps", 10, "BLOCK",
              actor_capabilities=["SCHEMA_WRITE", "ADMIN"]),
    ]
    engine = _make_engine(rules)

    # Only one capability — should NOT match BLOCK rule, falls through to ALLOW
    result = engine.evaluate("EXECUTION", "MUTATE", "SCHEMA", "ADMIN",
                             ["SCHEMA_WRITE"], {})
    assert result.verdict == PolicyVerdict.ALLOW

    # Both capabilities — matches BLOCK
    result = engine.evaluate("EXECUTION", "MUTATE", "SCHEMA", "ADMIN",
                             ["SCHEMA_WRITE", "ADMIN"], {})
    assert result.verdict == PolicyVerdict.BLOCK


def test_viewer_blocked_on_execution():
    """Mirrors the default policy viewer rule."""
    rules = [
        _rule("viewer-read-only", 40, "BLOCK",
              actor_role="VIEWER", intent_class="EXECUTION"),
    ]
    engine = _make_engine(rules)
    result = engine.evaluate("EXECUTION", "DEPLOY", "SERVICE", "VIEWER", [], {})
    assert result.verdict == PolicyVerdict.BLOCK


def test_unmatched_condition_falls_through():
    rules = [_rule("deploy-only", 10, "BLOCK", action_type="DEPLOY")]
    engine = _make_engine(rules)
    result = engine.evaluate("EXECUTION", "ROLLBACK", "SERVICE", "OPERATOR", [], {})
    assert result.verdict == PolicyVerdict.ALLOW


def test_dot_path_modify_nested():
    from xnch.policy.loader import RuleAction
    action = RuleAction(
        verdict="MODIFY",
        reason="force replicas",
        modify_spec={"field": "params.replicas", "value": 3},
    )
    rule = Rule(rule_id="max-replicas", priority=10,
                conditions=RuleConditions(),
                action=action)
    engine = _make_engine([rule])
    result = engine.evaluate("EXECUTION", "DEPLOY", "SERVICE", "OPERATOR", [],
                             {"params": {"replicas": 5}})
    assert result.modified_action_spec["params"]["replicas"] == 3
