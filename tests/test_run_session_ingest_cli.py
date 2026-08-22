"""Tests for the session-ingest CLI entrypoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from xnch.jobs import run_session_ingest as cli
from xnch.memory.session_ingest import IngestReport


@pytest.fixture
def mock_ingest(monkeypatch):
    mock = AsyncMock(return_value=IngestReport(succeeded=3))
    monkeypatch.setattr(cli, "ingest_sessions", mock)
    return mock


@pytest.fixture
def mock_stores(monkeypatch):
    calls: dict[str, object] = {}

    class FakePg:
        async def connect(self):
            calls["pg_connect"] = True

        async def close(self):
            calls["pg_close"] = True

    class FakeGraph:
        def __init__(self, *a, **k):
            pass

        def connect(self):
            calls["graph_connect"] = True

        def close(self):
            calls["graph_close"] = True

    monkeypatch.setattr(cli, "PgEpisodicStore", lambda *a, **k: FakePg())
    monkeypatch.setattr(cli, "GraphStore", lambda *a, **k: FakeGraph())
    return calls


async def test_backfill_uses_project_filter_by_default(
    mock_ingest, mock_stores, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    code = await cli.main(["--backfill"])
    assert code == 0
    kwargs = mock_ingest.await_args.kwargs
    assert kwargs["directories"] == [str(tmp_path)]
    assert kwargs["dry_run"] is False


async def test_all_flag_clears_project_filter(mock_ingest, mock_stores):
    await cli.main(["--backfill", "--all"])
    assert mock_ingest.await_args.kwargs["directories"] is None


async def test_dry_run_flag_passthrough(mock_ingest, mock_stores):
    await cli.main(["--backfill", "--dry-run"])
    assert mock_ingest.await_args.kwargs["dry_run"] is True


async def test_incremental_mode(mock_ingest, mock_stores):
    await cli.main(["--incremental"])
    assert mock_ingest.await_args is not None


async def test_dry_run_never_touches_kuzu(mock_ingest, mock_stores):
    await cli.main(["--backfill", "--dry-run"])
    assert "graph_connect" not in mock_stores


async def test_session_id_mode(mock_ingest, mock_stores):
    await cli.main(["--session-id", "ses_x"])
    assert mock_ingest.await_args.kwargs["session_ids"] == ["ses_x"]


async def test_since_flag_parses_iso(mock_ingest, mock_stores):
    await cli.main(["--backfill", "--since", "2026-08-01"])
    kwargs = mock_ingest.await_args.kwargs
    assert kwargs["updated_since_ms"] > 1_700_000_000_000


async def test_settings_project_dirs_override(
    mock_ingest, mock_stores, monkeypatch
):
    monkeypatch.setattr(cli.settings, "session_ingest_project_dirs", "/a,/b")
    await cli.main(["--backfill"])
    assert mock_ingest.await_args.kwargs["directories"] == ["/a", "/b"]
