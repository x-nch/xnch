"""Unit tests for HITL when-predicates and typed resume normalization."""
from __future__ import annotations

import pytest

from xnch.agents.hitl import (
    normalize_resume,
    parse_resume_decision,
    should_interrupt_execution,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("approve", True),
        ("reject", False),
        ({"decision": "approve"}, True),
        ({"decision": "reject"}, False),
        ({"approved": True}, True),
    ],
)
def test_normalize_resume(value: object, expected: bool) -> None:
    assert normalize_resume(value) is expected


def test_parse_resume_prefers_decision() -> None:
    assert parse_resume_decision(decision="approve", approved=False) is True
    assert parse_resume_decision(decision=None, approved=False) is False
    with pytest.raises(ValueError):
        parse_resume_decision()


def test_when_predicate_modes() -> None:
    assert should_interrupt_execution(intent_class="QUERY") is False
    assert should_interrupt_execution(intent_class="EXECUTION", mode="always") is True
    assert should_interrupt_execution(intent_class="EXECUTION", mode="never") is False

    low_risk = [{"composite_score": 0.95}]
    high_risk = [{"composite_score": 0.1, "risk_score": 0.9}]
    assert (
        should_interrupt_execution(
            intent_class="EXECUTION",
            evaluated=low_risk,
            mode="risk_threshold",
            risk_threshold=0.5,
        )
        is False
    )
    assert (
        should_interrupt_execution(
            intent_class="EXECUTION",
            evaluated=high_risk,
            mode="risk_threshold",
            risk_threshold=0.5,
        )
        is True
    )


def test_settings_flags() -> None:
    from xnch.config import Settings

    s = Settings()
    assert s.langgraph_pipeline is False
    assert s.hitl_execution_mode == "always"
    assert s.hitl_risk_threshold == 0.5
