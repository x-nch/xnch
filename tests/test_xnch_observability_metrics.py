"""Prometheus instrumentation: HTTP/HITL/consolidation metrics, deep health probes, scrape gating."""
from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver
from prometheus_client import REGISTRY

import xnch.agents.pipeline_graph as pg
from xnch.agents.pipeline_runtime import PipelineRuntime
from xnch.routes.pipeline import router as pipeline_router


def _get_sample_family(name: str):
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name == name:
                return family
    return None


def _sample_value(name: str, labels: dict[str, str] | None = None) -> float | None:
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name != name:
                continue
            if labels is None or all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return None


# ---------------------------------------------------------------------------
# Scrape-gating: host_allowed
# ---------------------------------------------------------------------------


class TestHostAllowed:
    def test_exact_host_allowed(self) -> None:
        from xnch.observability.metrics import host_allowed

        assert host_allowed("127.0.0.1", ["127.0.0.1"]) is True

    def test_cidr_match(self) -> None:
        from xnch.observability.metrics import host_allowed

        assert host_allowed("192.168.50.2", ["192.168.50.0/24"]) is True

    def test_public_host_denied(self) -> None:
        from xnch.observability.metrics import host_allowed

        assert host_allowed("203.0.113.9", ["127.0.0.1", "::1", "192.168.50.0/24"]) is False

    def test_garbage_host_denied(self) -> None:
        from xnch.observability.metrics import host_allowed

        assert host_allowed("not-an-ip-or-cidr", ["127.0.0.1"]) is False


# ---------------------------------------------------------------------------
# /metrics endpoint + HTTP middleware
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics_app(monkeypatch: pytest.MonkeyPatch):
    from unittest.mock import AsyncMock

    from xnch.config import settings

    monkeypatch.setattr(settings, "metrics_allow_cidrs", ["127.0.0.1", "::1"])
    from xnch.main import app as real_app

    real_app.state.kv_cache = AsyncMock()
    real_app.state.kv_cache.ping = AsyncMock(return_value=True)
    real_app.state.get_state_version = AsyncMock(return_value="v1")
    return real_app


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_request_series(metrics_app: FastAPI) -> None:
    transport = ASGITransport(app=metrics_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        health = await client.get("/health")
        assert health.status_code in (200, 503)

        resp = await client.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
    assert 'route="/health"' in body
    assert "xnch_http_requests_total" in body
    assert "xnch_http_request_seconds_count" in body


@pytest.mark.asyncio
async def test_metrics_denied_for_non_allowlisted_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xnch.config import settings

    monkeypatch.setattr(settings, "metrics_allow_cidrs", ["10.99.0.0/16"])
    from xnch.main import app as real_app

    transport = ASGITransport(app=real_app, client=("203.0.113.9", 1234))
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/metrics")
    assert resp.status_code == 403


def test_middleware_labels_status_and_route() -> None:
    app = FastAPI()

    from xnch.observability.metrics import install_metrics_middleware

    install_metrics_middleware(app)

    @app.get("/thing/{thing_id}")
    async def thing(thing_id: str) -> dict[str, str]:
        return {"id": thing_id}

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise RuntimeError("kaboom")

    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/thing/abc").status_code == 200
    assert client.post("/nope").status_code == 404
    assert client.get("/boom").status_code == 500

    assert _sample_value(
        "xnch_http_requests_total",
        {"method": "GET", "route": "/thing/{thing_id}", "status": "200"},
    ) >= 1.0
    assert _sample_value(
        "xnch_http_requests_total", {"method": "POST", "route": "unmatched", "status": "404"}
    ) >= 1.0
    assert _sample_value(
        "xnch_http_requests_total", {"method": "GET", "route": "/boom", "status": "500"}
    ) >= 1.0


# ---------------------------------------------------------------------------
# HITL gate metrics
# ---------------------------------------------------------------------------


class TestHitlMetricHelpers:
    def test_open_interrupt_increments_pending_and_age(self) -> None:
        from xnch.observability.metrics import (
            hitl_pending_interrupts,
            record_interrupt_opened,
        )

        tid = f"t-{uuid4()}"
        record_interrupt_opened(tid)
        try:
            assert _sample_value("xnch_hitl_interrupts_opened_total") >= 1.0
            assert hitl_pending_interrupts._value.get() >= 1
            age = _sample_value("xnch_hitl_oldest_pending_age_seconds")
            assert age is not None and 0.0 <= age < 5.0
        finally:
            from xnch.observability.metrics import record_decision

            record_decision(tid, "approve")

    def test_decision_counts_by_label_and_clears_pending(self) -> None:
        from xnch.observability.metrics import record_decision, record_interrupt_opened

        tid = f"t-{uuid4()}"
        rejected_before = _sample_value("xnch_hitl_decisions_total", {"decision": "rejected"}) or 0.0
        approved_before = _sample_value("xnch_hitl_decisions_total", {"decision": "approved"}) or 0.0
        record_interrupt_opened(tid)
        record_decision(tid, "reject")

        assert (_sample_value("xnch_hitl_decisions_total", {"decision": "rejected"}) or 0.0) == rejected_before + 1.0
        assert (_sample_value("xnch_hitl_decisions_total", {"decision": "approved"}) or 0.0) == approved_before

    def test_time_to_decision_observed_on_decide(self) -> None:
        import time as _time

        from xnch.observability.metrics import record_decision, record_interrupt_opened

        tid = f"t-{uuid4()}"
        count_before = _sample_value(
            "xnch_hitl_time_to_decision_seconds_count", {"decision": "approved"}
        ) or 0.0
        sum_before = _sample_value(
            "xnch_hitl_time_to_decision_seconds_sum", {"decision": "approved"}
        ) or 0.0

        record_interrupt_opened(tid)
        _time.sleep(0.01)
        record_decision(tid, "approve")

        assert (_sample_value("xnch_hitl_time_to_decision_seconds_count", {"decision": "approved"}) or 0.0) == count_before + 1.0
        assert (
            _sample_value("xnch_hitl_time_to_decision_seconds_sum", {"decision": "approved"}) or 0.0
        ) > sum_before

    def test_time_to_decision_skipped_when_thread_unknown(self) -> None:
        """After a restart the open-time is lost; decision still counts, TTD skipped."""
        from xnch.observability.metrics import record_decision

        count_before = _sample_value(
            "xnch_hitl_time_to_decision_seconds_count", {"decision": "rejected"}
        ) or 0.0
        record_decision(f"unknown-after-restart-{uuid4()}", "reject")
        assert (_sample_value("xnch_hitl_time_to_decision_seconds_count", {"decision": "rejected"}) or 0.0) == count_before


def _stub_pipeline_nodes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    selected = {
        "option_id": str(uuid4()),
        "action_type": "apply",
        "action_spec": {"type": "noop", "target": "x", "params": {}},
        "reversible": True,
        "estimated_side_effects": [],
    }

    async def classify_intent(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "intent": {
                "intent_class": "EXECUTION",
                "action_type": "apply",
                "target_entity_id": "e1",
                "target_entity_class": "service",
                "urgency": "normal",
                "ambiguity_score": 0.0,
                "raw_input": state["raw_input"],
            },
            "events": [],
        }

    async def assemble_context(state: dict[str, Any], **_stores: Any) -> dict[str, Any]:
        return {"context": {"system_prompt": "", "recent_turns": [], "relevant_episodes": [], "entity_context": [], "relationship_context": [], "perception_snippets": []}}

    async def generate_options(state: dict[str, Any]) -> dict[str, Any]:
        return {"options": [dict(selected)]}

    async def filter_policy(state: dict[str, Any]) -> dict[str, Any]:
        return {"policy_verdicts": [{"verdict": "ALLOW", "policy_refs": [], "warnings": [], "modified_action_spec": None}]}

    async def evaluate(state: dict[str, Any]) -> dict[str, Any]:
        return {"evaluated": [{**selected, "composite_score": 0.9}], "events": []}

    async def select(state: dict[str, Any]) -> dict[str, Any]:
        from langgraph.types import interrupt

        from xnch.agents.hitl import normalize_resume

        approved = normalize_resume(interrupt({"action": "approve_execution", "selected": state["options"][0], "intent": state["intent"]}))
        return {"selected": state["options"][0] if approved else None, "events": []}

    async def compile_plan(state: dict[str, Any]) -> dict[str, Any]:
        return {"compiled_plan": {"nodes": [{"action_type": "noop"}]}, "events": []}

    async def dispatch(state: dict[str, Any]) -> dict[str, Any]:
        return {"events": []}

    monkeypatch.setattr(pg, "classify_intent", classify_intent)
    monkeypatch.setattr(pg, "assemble_context", assemble_context)
    monkeypatch.setattr(pg, "generate_options", generate_options)
    monkeypatch.setattr(pg, "filter_policy", filter_policy)
    monkeypatch.setattr(pg, "evaluate", evaluate)
    monkeypatch.setattr(pg, "select", select)
    monkeypatch.setattr(pg, "compile_plan", compile_plan)
    monkeypatch.setattr(pg, "dispatch", dispatch)
    return selected


@pytest.fixture
async def hitl_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _stub_pipeline_nodes(monkeypatch)
    runtime = PipelineRuntime(checkpointer=MemorySaver())
    await runtime.start()
    app = FastAPI()
    app.include_router(pipeline_router)
    app.state.pipeline_runtime = runtime
    with TestClient(app) as client:
        yield client
    await runtime.stop()


def test_pipeline_invoke_and_resume_emit_hitl_metrics(hitl_client: TestClient) -> None:
    opened_before = _sample_value("xnch_hitl_interrupts_opened_total") or 0.0

    r = hitl_client.post(
        "/governance/pipeline/invoke",
        json={"session_id": str(uuid4()), "raw_input": "apply change", "thread_id": "obs-1"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "interrupted"
    assert (_sample_value("xnch_hitl_interrupts_opened_total") or 0.0) == opened_before + 1.0
    assert (_sample_value("xnch_hitl_pending_interrupts") or 0.0) >= 1.0

    approved_before = _sample_value(
        "xnch_hitl_decisions_total", {"decision": "approved"}
    ) or 0.0
    resume = hitl_client.post(
        "/governance/pipeline/resume", json={"thread_id": "obs-1", "decision": "approve"}
    )
    assert resume.status_code == 200
    assert (_sample_value("xnch_hitl_decisions_total", {"decision": "approved"})) == approved_before + 1.0
    assert (_sample_value("xnch_hitl_pending_interrupts") or 0.0) == 0.0


# ---------------------------------------------------------------------------
# Deep memory-tier health probes
# ---------------------------------------------------------------------------


@pytest.fixture
async def fake_redis() -> fakeredis.aioredis.FakeRedis:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


async def test_redis_canary_passes_with_real_ttl(fake_redis) -> None:
    from xnch.observability.deep_health import check_redis_ttl_canary

    result = await check_redis_ttl_canary(fake_redis)
    assert result.ok is True
    assert result.latency_ms >= 0.0
    assert "ttl" in result.detail.lower()


async def test_redis_canary_fails_when_ttl_missing(fake_redis) -> None:
    from xnch.observability.deep_health import check_redis_ttl_canary

    await fake_redis.set("obs:canary-sentinel", "pre-existing-no-ttl")
    result = await check_redis_ttl_canary(fake_redis)
    assert result.ok is False


class _FakePool:
    """Minimal asyncpg.Pool stand-in returning canned values."""

    def __init__(self, value: int = 3, exc: Exception | None = None) -> None:
        self._value = value
        self._exc = exc
        self.last_sql: str | None = None

    class _Conn:
        def __init__(self, outer: "_FakePool") -> None:
            self._outer = outer

        async def fetchval(self, sql: str) -> int:
            self._outer.last_sql = sql
            if self._outer._exc:
                raise self._outer._exc
            return self._outer._value

    def acquire(self):
        outer = self

        class _Ctx:
            async def __aenter__(self):
                return _FakePool._Conn(outer)

            async def __aexit__(self, *exc_info: object) -> None:
                return None

        return _Ctx()


async def test_postgres_probe_runs_real_episodic_query() -> None:
    from xnch.observability.deep_health import check_postgres_episodic

    pool = _FakePool(value=7)
    result = await check_postgres_episodic(pool)
    assert result.ok is True
    assert pool.last_sql is not None and "episodes" in pool.last_sql


async def test_postgres_probe_fails_on_error() -> None:
    from xnch.observability.deep_health import check_postgres_episodic

    pool = _FakePool(exc=RuntimeError("connection refused"))
    result = await check_postgres_episodic(pool)
    assert result.ok is False


async def test_kuzu_probe_roundtrip_write_and_read(tmp_path) -> None:
    from xnch.memory.graph_store import GraphStore
    from xnch.memory.relationship_store import RelationshipStore
    from xnch.observability.deep_health import check_kuzu_roundtrip

    rs = RelationshipStore.__new__(RelationshipStore)
    store = GraphStore(db_path=tmp_path, relationship_store=rs)
    store.connect()
    try:
        result = await check_kuzu_roundtrip(store)
        assert result.ok is True
        fetched = store.get_entity_by_name("_obs_probe")
        assert fetched is not None
    finally:
        store.close()


async def test_deep_health_loop_updates_tier_gauges(fake_redis, tmp_path) -> None:
    from xnch.observability.deep_health import DeepHealthRunner

    runner = DeepHealthRunner(redis_client=fake_redis, pg_pool=_FakePool(), graph_store=None, interval_s=0.05)
    await runner.run_once()
    assert _sample_value("xnch_memory_tier_up", {"tier": "redis"}) == 1.0
    assert _get_sample_family("xnch_memory_tier_probe_seconds") is not None


# ---------------------------------------------------------------------------
# SQLite store latency
# ---------------------------------------------------------------------------


async def test_timed_sqlite_records_store_latency(tmp_path) -> None:
    from xnch.memory.db import get_state_version, init_db
    from xnch.observability.metrics import timed_sqlite

    @timed_sqlite("probe_store")
    async def probe() -> str:
        return await get_state_version(tmp_path / "probe.db")

    count_before = _sample_value("xnch_sqlite_query_seconds_count", {"store": "probe_store"}) or 0.0

    db = tmp_path / "probe.db"
    await init_db(db)
    assert await probe() == "v1"

    assert (
        _sample_value("xnch_sqlite_query_seconds_count", {"store": "probe_store"}) or 0.0
    ) == count_before + 1.0


# ---------------------------------------------------------------------------
# Consolidation job timing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consolidate_endpoint_records_success_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    import xnch.routes.admin as admin_mod
    from xnch.routes.admin import router as admin_router

    async def fake_run(**kwargs: Any) -> dict[str, int]:
        await asyncio.sleep(0.01)
        return {"extraction_failures": 0, "decayed": 5}

    monkeypatch.setattr(admin_mod, "run_consolidation", fake_run)

    app = FastAPI()
    app.include_router(admin_router)
    state = MagicMock()
    state.pg_episodic = AsyncMock()
    state.relationship_store = AsyncMock()
    state.graph_store = AsyncMock()
    app.state = state

    before = _sample_value("xnch_consolidation_runs_total", {"result": "success"}) or 0.0
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/admin/consolidate")
    assert resp.status_code == 200
    assert _sample_value("xnch_consolidation_runs_total", {"result": "success"}) == before + 1.0
    assert (_sample_value("xnch_consolidation_run_seconds_count") or 0.0) >= 1.0


@pytest.mark.asyncio
async def test_consolidate_endpoint_records_failure_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    import asyncio

    import xnch.routes.admin as admin_mod
    from xnch.routes.admin import router as admin_router

    async def fake_run(**kwargs: Any) -> dict[str, int]:
        raise RuntimeError("pg exploded")

    monkeypatch.setattr(admin_mod, "run_consolidation", fake_run)

    app = FastAPI()
    app.include_router(admin_router)
    state = MagicMock()
    state.pg_episodic = AsyncMock()
    state.relationship_store = AsyncMock()
    state.graph_store = AsyncMock()
    app.state = state

    before = _sample_value("xnch_consolidation_runs_total", {"result": "failure"}) or 0.0
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/admin/consolidate")
    assert resp.status_code == 500
    assert _sample_value("xnch_consolidation_runs_total", {"result": "failure"}) == before + 1.0
