"""Tests for the session-ingest orchestrator (fake stores, fixture SQLite)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xnch.memory.session_ingest.ingestor import ingest_sessions
from xnch.memory.session_ingest.models import (
    FactEntity,
    FactTriple,
    SessionDigest,
    SessionSummary,
)

MS0 = 1_785_000_000_000
SECRET_KEY = "sk-proj-ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "opencode.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE session (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, parent_id TEXT,
            slug TEXT NOT NULL, directory TEXT NOT NULL, title TEXT NOT NULL,
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
    rows = [
        ("ses_a", "/repo", "Fix auth bug"),
        ("ses_b", "/repo", "Refactor router"),
        ("ses_c", "/elsewhere", "Other project"),
    ]
    for sid, directory, title in rows:
        conn.execute(
            "INSERT INTO session VALUES (?,?,?,?,?,?,?,?,'build','ornith')",
            (sid, "p", None, sid, directory, title, MS0, MS0 + 60_000),
        )
        mid = f"m_{sid}"
        conn.execute(
            "INSERT INTO message VALUES (?,?,?,?)",
            (mid, sid, MS0 + 10, '{"role": "user"}'),
        )
        conn.execute(
            "INSERT INTO part VALUES (?,?,?,?,?)",
            (
                f"p_{sid}",
                mid,
                sid,
                MS0 + 11,
                f'{{"type": "text", "text": "work on {title} using {SECRET_KEY}"}}',
            ),
        )
    conn.commit()
    conn.close()
    return db_path


class FakePg:
    def __init__(self) -> None:
        self.episodes: list[dict] = []
        self.done: set[str] = set()
        self.failed: dict[str, str] = {}

    async def ledger_completed_ids(self) -> set[str]:
        return set(self.done)

    async def store_session_episode(self, **kwargs) -> str:
        eid = f"ep-{len(self.episodes) + 1}"
        self.episodes.append(kwargs)
        return eid

    async def ledger_mark_done(self, session_id, episode_id, facts_count=0) -> None:
        self.done.add(session_id)

    async def ledger_mark_failed(self, session_id, error) -> None:
        self.failed[session_id] = error


class FakeGraph:
    def __init__(self) -> None:
        self.entities: list[tuple] = []
        self.relations: list[dict] = []

    def upsert_entity(self, id, name, type_) -> None:
        self.entities.append((id, name, type_))

    async def upsert_relation(self, **kwargs) -> None:
        self.relations.append(kwargs)


def _ok_summarizer(marker: str = ""):
    async def summarize(digest: SessionDigest) -> SessionSummary:
        return SessionSummary(
            summary=f"summary of {digest.session_id} {marker}",
            decisions=[f"decision {digest.session_id}"],
            outcome="success",
            facts=[
                FactTriple(
                    subject=FactEntity(id="kuzu", name="Kuzu", type="technology"),
                    relation="chosen_over",
                    object=FactEntity(id="memgraph", name="Memgraph", type="technology"),
                )
            ],
        )

    return summarize


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return _make_db(tmp_path)


async def test_happy_path_stores_episode_and_bi_temporal_facts(db_path):
    pg, graph = FakePg(), FakeGraph()
    report = await ingest_sessions(
        db_path, pg, graph, directories=["/repo"], summarize=_ok_summarizer()
    )
    assert (report.succeeded, report.failed, report.skipped) == (2, 0, 0)
    assert len(pg.episodes) == 2
    assert len(graph.relations) == 2
    rel = graph.relations[0]
    assert rel["source"].startswith("opencode:ses_")
    assert rel["valid_from"] == datetime.fromtimestamp(
        (MS0 + 60_000) / 1000, tz=timezone.utc
    )


async def test_secrets_redacted_before_any_write(db_path):
    pg, graph = FakePg(), FakeGraph()
    await ingest_sessions(db_path, pg, graph, summarize=_ok_summarizer())
    for ep in pg.episodes:
        blob = str(ep)
        assert SECRET_KEY not in blob
        assert "[REDACTED:" in blob
    for rel in graph.relations:
        assert SECRET_KEY not in str(rel)


async def test_double_ingestion_is_idempotent(db_path):
    pg, graph = FakePg(), FakeGraph()
    await ingest_sessions(db_path, pg, graph, summarize=_ok_summarizer())
    first_count = len(pg.episodes)
    second = await ingest_sessions(
        db_path, pg, graph, summarize=_ok_summarizer()
    )
    assert len(pg.episodes) == first_count
    assert second.succeeded == 0
    assert second.skipped == 3


async def test_summarizer_failure_marks_failed_and_continues(db_path):
    pg, graph = FakePg(), FakeGraph()

    async def flaky(digest: SessionDigest) -> SessionSummary:
        if digest.session_id == "ses_a":
            raise ConnectionError("ornith down")
        return (await _ok_summarizer()(digest))

    report = await ingest_sessions(db_path, pg, graph, summarize=flaky)
    assert report.failed == 1 and report.succeeded == 2
    assert "ses_a" in pg.failed
    retry_report = await ingest_sessions(db_path, pg, graph, summarize=_ok_summarizer())
    assert retry_report.succeeded == 1


async def test_dry_run_writes_nothing(db_path):
    pg, graph = FakePg(), FakeGraph()
    report = await ingest_sessions(
        db_path, pg, graph, summarize=_ok_summarizer(), dry_run=True
    )
    assert report.succeeded == 3
    assert pg.episodes == [] and graph.relations == []
    assert pg.done == set()


async def test_limit_caps_processed_sessions(db_path):
    pg, graph = FakePg(), FakeGraph()
    report = await ingest_sessions(
        db_path, pg, graph, summarize=_ok_summarizer(), limit=1
    )
    assert report.scanned == 1 and report.succeeded == 1
    assert len(pg.episodes) == 1


async def test_session_ids_narrow_selection(db_path):
    pg, graph = FakePg(), FakeGraph()
    report = await ingest_sessions(
        db_path, pg, graph, session_ids=["ses_c"], summarize=_ok_summarizer()
    )
    assert report.succeeded == 1
    assert [ep["raw_text"] for ep in pg.episodes]
    assert graph.relations[0]["source"] == "opencode:ses_c"
