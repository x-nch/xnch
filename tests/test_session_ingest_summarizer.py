"""Tests for the ornith-backed session summarizer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from xnch.memory.session_ingest.models import SessionDigest
from xnch.memory.session_ingest import summarizer as smod
from xnch.memory.session_ingest.summarizer import summarize_session


def _digest() -> SessionDigest:
    return SessionDigest(
        session_id="ses_x",
        title="Migrate graph store",
        directory="/repo",
        started_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 8, 1, 2, tzinfo=timezone.utc),
        goal="Decide between Kuzu and Memgraph",
        files_touched=["xnch/memory/graph_store.py"],
        tools_used={"edit": 2},
        transcript_digest="user: pick a store\n$ pytest -q\nassistant: chose Kuzu",
    )


def _llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage = MagicMock(total_tokens=42)
    return resp


@pytest.fixture
def acompletion(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(smod.litellm, "acompletion", mock)
    return mock


@pytest.fixture
def trace_spy(monkeypatch):
    spy = AsyncMock()
    monkeypatch.setattr(smod, "trace_llm_call", spy)
    return spy


async def test_plain_json_payload(acompletion, trace_spy):
    payload = {"summary": "s", "decisions": ["d"], "outcome": "success", "facts": []}
    acompletion.return_value = _llm_response(json.dumps(payload))
    result = await summarize_session(_digest())
    assert result.summary == "s"
    assert result.decisions == ["d"]
    trace_spy.assert_awaited_once()


async def test_fenced_json_payload(acompletion, trace_spy):
    payload = '{"summary": "fenced", "decisions": [], "outcome": null, "facts": []}'
    acompletion.return_value = _llm_response(f"```json\n{payload}\n```")
    result = await summarize_session(_digest())
    assert result.summary == "fenced"


async def test_prose_wrapped_bracket_rescue(acompletion, trace_spy):
    payload = (
        '{"summary": "rescued", "decisions": [], "outcome": "ok", "facts": '
        '[{"subject": {"id": "kuzu", "name": "Kuzu", "type": "technology"},'
        ' "relation": "chosen_over",'
        ' "object": {"id": "memgraph", "name": "Memgraph", "type": "technology"}}]}'
    )
    acompletion.return_value = _llm_response(f"Here you go:\n{payload}\nDone.")
    result = await summarize_session(_digest())
    assert result.summary == "rescued"
    assert result.facts[0].relation == "chosen_over"


async def test_routes_through_opencode_go_endpoint(acompletion, trace_spy):
    acompletion.return_value = _llm_response('{"summary": "x"}')
    await summarize_session(_digest())
    kwargs = acompletion.await_args.kwargs
    assert kwargs["model"] == smod._model_name()
    assert kwargs["api_base"] == smod.settings.opencode_go_api_url
    assert kwargs["api_base"] != ""
    assert kwargs.get("api_key") == smod.settings.opencode_go_api_key


async def test_prompt_contains_goal_transcript_and_files(acompletion, trace_spy):
    acompletion.return_value = _llm_response('{"summary": "x"}')
    await summarize_session(_digest())
    user_msg = acompletion.await_args.kwargs["messages"][1]["content"]
    assert "Decide between Kuzu and Memgraph" in user_msg
    assert "$ pytest -q" in user_msg
    assert "xnch/memory/graph_store.py" in user_msg


async def test_unparseable_payload_raises(acompletion, trace_spy):
    acompletion.return_value = _llm_response("not json at all")
    with pytest.raises(ValueError):
        await summarize_session(_digest())


async def test_trace_includes_session_scoped_trace_id(acompletion, trace_spy):
    acompletion.return_value = _llm_response('{"summary": "x"}')
    await summarize_session(_digest())
    assert trace_spy.await_args.kwargs["trace_id"] == "session-ingest:ses_x"


async def test_llm_transport_error_propagates(acompletion, trace_spy):
    acompletion.side_effect = ConnectionError("litellm down")
    with pytest.raises(ConnectionError):
        await summarize_session(_digest())
