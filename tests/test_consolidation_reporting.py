"""Consolidation reporting: extraction counts surfaced instead of swallowed."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import xnch.jobs.consolidation as cons
import xnch.memory.graph_extractor as gmod
from xnch.routes.admin import router as admin_router

TRIPLE_A = {
    "subject": {"id": "a", "name": "A", "type": "svc"},
    "relation": "uses",
    "object": {"id": "b", "name": "B", "type": "svc"},
}


def _ep(id_: str, text: str | None) -> dict:
    return {"id": id_, "raw_text": text, "summary": None}


async def test_extract_and_store_counts_and_skip_retry(monkeypatch):
    """Batch continues past a failing episode and reports per-outcome counts."""
    async def fake_extract(text: str) -> list[dict]:
        if text == "bad":
            raise RuntimeError("llm down")
        return [TRIPLE_A]

    monkeypatch.setattr(gmod, "_extract_triples", fake_extract)
    episodes = [
        _ep("ok1", "good"),
        _ep("bad1", "bad"),
        _ep("empty", None),
    ]
    pg = SimpleNamespace(
        fetch_unextracted_for_graph=AsyncMock(return_value=episodes),
        mark_graph_extracted=AsyncMock(),
    )
    graph = MagicMock()
    graph.upsert_entity.return_value = None
    graph.upsert_relation = AsyncMock(return_value=None)

    out = await gmod.extract_and_store(
        pg_episodic=pg, relationship_store=None, graph_store=graph
    )

    assert out["triples_written"] == 1
    assert out["episodes_processed"] == 2
    assert out["extraction_failures"] == 1
    marked = sorted(pg.mark_graph_extracted.await_args.args[0])
    assert marked == ["empty", "ok1"]


async def test_run_consolidation_returns_counts(monkeypatch):
    async def fake_extract(**kwargs):
        return {"triples_written": 5, "episodes_processed": 4, "extraction_failures": 1}

    monkeypatch.setattr(cons, "extract_and_store", fake_extract)
    pg = SimpleNamespace(
        fetch_episodes_for_decay=AsyncMock(return_value=[]),
        apply_decay_batch=AsyncMock(),
        close=AsyncMock(),
    )
    out = await cons.run_consolidation(pg_episodic=pg)
    assert out == {
        "triples_written": 5,
        "episodes_processed": 4,
        "extraction_failures": 1,
        "archived": 0,
    }


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(admin_router)
    app.state.pg_episodic = object()
    app.state.relationship_store = object()
    app.state.graph_store = object()
    return app


def test_admin_consolidate_reports_failures(monkeypatch):
    counts = {
        "triples_written": 0,
        "episodes_processed": 5,
        "extraction_failures": 3,
        "archived": 0,
    }
    async def fake_rc(**kwargs):
        return counts

    monkeypatch.setattr("xnch.routes.admin.run_consolidation", fake_rc)
    client = TestClient(_app())
    r = client.post("/admin/consolidate")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] != "ok"
    assert body["extraction_failures"] == 3


def test_admin_consolidate_ok_when_no_failures(monkeypatch):
    async def fake_rc(**kwargs):
        return {
            "triples_written": 7,
            "episodes_processed": 6,
            "extraction_failures": 0,
            "archived": 2,
        }

    monkeypatch.setattr("xnch.routes.admin.run_consolidation", fake_rc)
    client = TestClient(_app())
    body = client.post("/admin/consolidate").json()
    assert body["status"] == "ok"
    assert body["triples_written"] == 7
    assert body["archived"] == 2
