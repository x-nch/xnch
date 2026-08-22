"""Tests for the scheduled session-ingest job wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
from unittest.mock import AsyncMock

import xnch.main as xnch_main
from xnch.config import settings
from xnch.jobs import session_ingest as jobmod
from xnch.jobs.session_ingest import run_incremental_ingest
from xnch.memory.session_ingest import IngestReport


class FakePg:
    pass


class FakeGraph:
    pass


@pytest.fixture
def mock_ingest(monkeypatch):
    mock = AsyncMock(return_value=IngestReport(succeeded=2))
    monkeypatch.setattr(jobmod, "ingest_sessions", mock)
    return mock


@pytest.fixture
def fake_db(monkeypatch, tmp_path):
    db = tmp_path / "opencode.db"
    db.write_bytes(b"")
    monkeypatch.setattr(jobmod.settings, "session_ingest_db_path", str(db))
    return db


async def test_passes_live_stores_and_dry_run_false(
    mock_ingest, fake_db
):
    pg, graph = FakePg(), FakeGraph()
    await run_incremental_ingest(pg_episodic=pg, graph_store=graph)
    args, kwargs = mock_ingest.await_args
    assert args[0] == Path(fake_db)
    assert args[1] is pg and args[2] is graph
    assert kwargs["dry_run"] is False


async def test_missing_db_returns_none_without_calling(mock_ingest, tmp_path, monkeypatch):
    monkeypatch.setattr(
        jobmod.settings, "session_ingest_db_path", str(tmp_path / "nope.db")
    )
    result = await run_incremental_ingest(pg_episodic=FakePg(), graph_store=FakeGraph())
    assert result is None
    mock_ingest.assert_not_awaited()


async def test_settings_project_dirs_flow_through(mock_ingest, fake_db, monkeypatch):
    monkeypatch.setattr(
        jobmod.settings, "session_ingest_project_dirs", "/repo/a,/repo/b"
    )
    await run_incremental_ingest(pg_episodic=FakePg(), graph_store=FakeGraph())
    assert mock_ingest.await_args.kwargs["directories"] == ["/repo/a", "/repo/b"]


async def test_no_dirs_configured_passes_none(mock_ingest, fake_db, monkeypatch):
    monkeypatch.setattr(jobmod.settings, "session_ingest_project_dirs", "")
    await run_incremental_ingest(pg_episodic=FakePg(), graph_store=FakeGraph())
    assert mock_ingest.await_args.kwargs["directories"] is None


def test_lifespan_registers_job_when_enabled(monkeypatch):
    assert 'id="session_ingest"' in Path(xnch_main.__file__).read_text()
