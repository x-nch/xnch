"""Per-agent spend-budget gate tests."""

from xnch.policies.policy_filter import check_agent_budget


def test_over_budget_blocked() -> None:
    assert check_agent_budget("finance", 1000, 0.09) is False


def test_under_budget_allowed() -> None:
    assert check_agent_budget("finance", 1000, 0.01) is True
