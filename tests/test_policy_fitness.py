"""Policy rule fitness — outcome consistency tests."""
from xnch.learning.evolution.policy_fitness import compute_rule_fitness, _outcome_score
from xnch.learning.evolution.policy_rule import PolicyRule


def _episode(intent_class, action_type, entity_class, actor_role, outcome):
    return {
        "intent_class": intent_class,
        "action_type": action_type,
        "entity_class": entity_class,
        "actor_role": actor_role,
        "outcome": outcome,
    }


def test_allow_rule_fitness_high_when_matching_episodes_succeed():
    rule = PolicyRule(conditions={"intent_class": "EXECUTION"}, verdict="ALLOW")
    episodes = [
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "SUCCESS"),
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "SUCCESS"),
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "SUCCESS"),
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "SUCCESS"),
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "SUCCESS"),
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "FAILURE"),
    ]
    fitness = compute_rule_fitness(rule, episodes)
    assert fitness > 0.5  # 5/6 success favors ALLOW


def test_block_rule_fitness_high_when_matching_episodes_fail():
    rule = PolicyRule(conditions={"intent_class": "EXECUTION"}, verdict="BLOCK")
    episodes = [
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "FAILURE"),
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "FAILURE"),
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "FAILURE"),
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "FAILURE"),
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "FAILURE"),
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "SUCCESS"),
    ]
    fitness = compute_rule_fitness(rule, episodes)
    assert fitness > 0.5  # 5/6 failure favors BLOCK


def test_rule_with_no_matching_episodes_gets_neutral_fitness():
    rule = PolicyRule(conditions={"intent_class": "EXECUTION"}, verdict="ALLOW")
    episodes = [
        _episode("QUERY", "LIST", "FILE", "viewer", "SUCCESS"),
    ]
    assert compute_rule_fitness(rule, episodes) == 0.5


def test_partial_outcome_scores_partially():
    assert _outcome_score("SUCCESS") == 1.0
    assert _outcome_score("FAILURE") == 0.0
    assert _outcome_score("PARTIAL") == 0.5
    assert _outcome_score("UNKNOWN") == 0.5


def test_fitness_needs_min_observations():
    rule = PolicyRule(conditions={"intent_class": "EXECUTION"}, verdict="ALLOW")
    episodes = [
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "SUCCESS"),
    ]
    # Only 1 observation → below the minimum → neutral
    assert compute_rule_fitness(rule, episodes) == 0.5


def test_wildcard_rule_matches_all_for_fitness():
    rule = PolicyRule(conditions={}, verdict="ALLOW")
    episodes = [
        _episode("EXECUTION", "DEPLOY", "SERVICE", "operator", "SUCCESS"),
        _episode("QUERY", "LIST", "FILE", "viewer", "SUCCESS"),
        _episode("QUERY", "LIST", "FILE", "viewer", "SUCCESS"),
        _episode("QUERY", "LIST", "FILE", "viewer", "SUCCESS"),
        _episode("DECISION", "PLAN", "SCHEMA", "agent", "FAILURE"),
        _episode("DECISION", "PLAN", "SCHEMA", "agent", "FAILURE"),
    ]
    fitness = compute_rule_fitness(rule, episodes)
    assert fitness > 0.5  # 4/6 success overall favors ALLOW
