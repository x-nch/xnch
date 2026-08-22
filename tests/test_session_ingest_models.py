"""Tests for session-ingest data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from xnch.memory.session_ingest.models import (
    FactEntity,
    FactTriple,
    SessionDigest,
    SessionSummary,
)


def _digest_kwargs() -> dict:
    return {
        "session_id": "ses_abc",
        "title": "Fix evaluator bug",
        "directory": "/Users/xnch/xnchSystems-ocReview",
        "project_id": "proj_1",
        "started_at": "2026-08-01T10:00:00+00:00",
        "ended_at": "2026-08-01T11:30:00+00:00",
        "goal": "Fix the evaluator regression in decision scoring",
        "files_touched": ["nexi/pipeline/evaluator.py"],
        "tools_used": {"bash": 12, "edit": 3},
        "transcript_digest": "user: fix this\nassistant: done",
    }


def test_session_digest_requires_identity_fields():
    with pytest.raises(ValidationError):
        SessionDigest()  # type: ignore[call-arg]


def test_session_digest_defaults():
    d = SessionDigest(**_digest_kwargs())
    assert d.agent == ""
    assert d.model == ""
    assert d.files_touched == ["nexi/pipeline/evaluator.py"]
    assert isinstance(d.tools_used, dict)


def test_fact_triple_validates_entity_shape():
    t = FactTriple.model_validate(
        {
            "subject": {"id": "kuzu", "name": "Kuzu", "type": "technology"},
            "relation": "chosen_over",
            "object": {"id": "memgraph", "name": "Memgraph", "type": "technology"},
        }
    )
    assert t.subject.name == "Kuzu"
    assert t.object.id == "memgraph"


def test_fact_triple_rejects_missing_relation():
    with pytest.raises(ValidationError):
        FactTriple(
            subject=FactEntity(id="a", name="A", type="entity"),
            object=FactEntity(id="b", name="B", type="entity"),
        )


def test_session_summary_defaults_and_roundtrip():
    s = SessionSummary(summary="Worked on evaluator fix.", decisions=["kept pipeline order"])
    assert s.outcome is None
    assert s.facts == []
    payload = s.model_dump(mode="json")
    assert SessionSummary.model_validate(payload).summary == s.summary
