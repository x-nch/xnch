"""Tests for the OpenCode SQLite session parser."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from xnch.memory.session_ingest.parser import (
    SessionRef,
    iter_session_refs,
    parse_session,
)

MS_MIN = 1_785_000_000_000


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "opencode.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE session (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, parent_id TEXT,
            slug TEXT NOT NULL,
            directory TEXT NOT NULL, title TEXT NOT NULL,
            time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
            agent TEXT, model TEXT);
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            time_created INTEGER NOT NULL, data TEXT NOT NULL);
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT NOT NULL, session_id TEXT NOT NULL,
            time_created INTEGER NOT NULL, data TEXT NOT NULL);
        """
    )
    conn.execute(
        "INSERT INTO session VALUES ('ses_a','p1',NULL,'slug-a',"
        "'/Users/xnch/xnchSystems-ocReview',"
        " 'Fix evaluator bug', ?, ?, 'build','ornith')",
        (MS_MIN, MS_MIN + 90 * 60 * 1000),
    )
    conn.execute(
        "INSERT INTO session VALUES ('ses_b','p2',NULL,'slug-b','/Users/xnch/other',"
        " 'Spotify session', ?, ?, 'build','ornith')",
        (MS_MIN + 1000, MS_MIN + 2000),
    )

    def msg(mid: str, sid: str, offset_ms: int, role: str) -> None:
        conn.execute(
            "INSERT INTO message VALUES (?,?,?,?)",
            (mid, sid, MS_MIN + offset_ms, json.dumps({"role": role})),
        )

    def part(pid: str, mid: str, sid: str, offset_ms: int, payload: dict) -> None:
        conn.execute(
            "INSERT INTO part VALUES (?,?,?,?,?)",
            (pid, mid, sid, MS_MIN + offset_ms, json.dumps(payload)),
        )

    msg("m1", "ses_a", 10, "user")
    part("pa1", "m1", "ses_a", 11, {"type": "text", "text": "Fix the evaluator regression"})
    msg("m2", "ses_a", 20, "assistant")
    part("pa2", "m2", "ses_a", 21, {"type": "text", "text": "Patched scoring path."})
    part("pa3", "m2", "ses_a", 22, {"type": "reasoning", "text": "thinking..."})
    part(
        "pa4",
        "m2",
        "ses_a",
        23,
        {
            "type": "tool",
            "tool": "bash",
            "state": {"status": "completed", "input": {"command": "pytest -q"}},
        },
    )
    part("pa5", "m2", "ses_a", 24, {"type": "step-start"})
    msg("m3", "ses_a", 30, "assistant")
    part(
        "pa6",
        "m3",
        "ses_a",
        31,
        {
            "type": "tool",
            "tool": "edit",
            "state": {
                "status": "completed",
                "input": {"filePath": "/repo/nexi/pipeline/evaluator.py"},
            },
        },
    )
    part(
        "pa7",
        "m3",
        "ses_a",
        32,
        {
            "type": "tool",
            "tool": "read",
            "state": {"status": "error", "input": {"filePath": "/repo/missing.py"}},
        },
    )
    part(
        "pa8",
        "m3",
        "ses_a",
        33,
        {
            "type": "tool",
            "tool": "read",
            "state": {"status": "completed", "input": {"filePath": "/repo/nexi/config.py"}},
        },
    )
    msg("m9", "ses_b", 40, "user")
    part("pb1", "m9", "ses_b", 41, {"type": "text", "text": "play something"})
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return _make_db(tmp_path)


def test_parse_extracts_times_and_identity(db_path: Path):
    d = parse_session(db_path, "ses_a")
    assert d is not None
    assert d.session_id == "ses_a"
    assert d.title == "Fix evaluator bug"
    assert d.model == "ornith"
    assert d.started_at.timestamp() * 1000 == pytest.approx(MS_MIN)
    assert d.ended_at is not None and d.ended_at.timestamp() * 1000 == pytest.approx(
        MS_MIN + 90 * 60 * 1000
    )


def test_goal_prefers_first_user_text_over_title(db_path: Path):
    d = parse_session(db_path, "ses_a")
    assert d is not None
    assert d.goal == "Fix the evaluator regression"


def test_files_touched_from_edit_and_read_inputs_deduped(db_path: Path):
    d = parse_session(db_path, "ses_a")
    assert d is not None
    assert d.files_touched == [
        "/repo/nexi/pipeline/evaluator.py",
        "/repo/missing.py",
        "/repo/nexi/config.py",
    ]


def test_tools_used_counts_all_tool_parts(db_path: Path):
    d = parse_session(db_path, "ses_a")
    assert d is not None
    assert d.tools_used == {"bash": 1, "edit": 1, "read": 2}


def test_transcript_has_user_assistant_and_commands_not_reasoning(db_path: Path):
    d = parse_session(db_path, "ses_a")
    assert d is not None
    t = d.transcript_digest
    assert "user: Fix the evaluator regression" in t
    assert "assistant: Patched scoring path." in t
    assert "$ pytest -q" in t
    assert "thinking..." not in t
    assert "step-start" not in t


def test_missing_session_returns_none(db_path: Path):
    assert parse_session(db_path, "ses_nope") is None


def test_iter_filters_by_directory_and_since(db_path: Path):
    refs = iter_session_refs(db_path)
    assert [r.session_id for r in refs] == ["ses_a", "ses_b"]
    filtered = iter_session_refs(db_path, directories=["/Users/xnch/xnchSystems"])
    assert [r.session_id for r in filtered] == ["ses_a"]
    recent = iter_session_refs(db_path, updated_since_ms=MS_MIN + 2001)
    assert [r.session_id for r in recent] == ["ses_a"]
    boundary = iter_session_refs(db_path, updated_since_ms=MS_MIN + 2000)
    assert [r.session_id for r in boundary] == ["ses_a", "ses_b"]


def test_iter_returns_refs_with_metadata(db_path: Path):
    ref = iter_session_refs(db_path)[0]
    assert isinstance(ref, SessionRef)
    assert ref.title == "Fix evaluator bug"
