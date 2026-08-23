"""Observability JSON surfaces (/observability/*) summarizing Prometheus for the operator UI."""
from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


def _prom_reply(result: list[dict[str, Any]]) -> dict[str, Any]:
    return {"status": "success", "data": {"resultType": "vector", "result": result}}


def _prom_matrix(series: list[list[Any]]) -> dict[str, Any]:
    return {"status": "success", "data": {"resultType": "matrix", "result": series}}


def _sample(metric: dict[str, str], val: float) -> dict[str, Any]:
    return {"metric": metric, "value": [time.time(), str(val)]}


class _StubTransport(httpx.AsyncBaseTransport):
    """Matches instant queries by expr substring; range queries via __RANGE__ prefix."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self._routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        expr = request.url.params.get("query", "")
        is_range = request.url.path.endswith("/api/v1/query_range")
        for needle, canned in self._routes.items():
            if not needle:
                continue
            if is_range != needle.startswith("__RANGE__"):
                continue
            clean = needle.removeprefix("__RANGE__")
            if clean in expr:
                if canned is None:
                    return httpx.Response(503)
                if is_range:
                    return httpx.Response(
                        200,
                        json={"status": "success", "data": {"resultType": "matrix", "result": canned}},
                    )
                return httpx.Response(
                    200,
                    json={"status": "success", "data": {"resultType": "vector", "result": canned}},
                )
        empty = {"resultType": "matrix" if is_range else "vector", "result": []}
        return httpx.Response(200, json={"status": "success", "data": empty})


def _build_app(prom_routes: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    from collections import deque
    from types import SimpleNamespace

    from xnch.config import settings
    from xnch.observability.prom_summary import PrometheusClient
    from xnch.routes.observability import router as obs_router

    monkeypatch.setattr(settings, "prometheus_url", "http://prom.test:9090")
    app = FastAPI()
    app.include_router(obs_router)
    app.state.prom_client = PrometheusClient(
        base_url=settings.prometheus_url,
        http_client=httpx.AsyncClient(
            base_url=settings.prometheus_url, transport=_StubTransport(prom_routes)
        ),
    )
    app.state.recent_alerts = deque(
        [
            {
                "status": "firing",
                "labels": {"alertname": "HitlGateBypassFiring", "severity": "security"},
                "annotations": {},
                "received_at": 1234.5,
            }
        ]
    )
    dh = SimpleNamespace()
    dh.last_results = {
        "redis": SimpleNamespace(ok=True, latency_ms=1.2, detail="ttl=2"),
        "postgres": SimpleNamespace(ok=True, latency_ms=3.4, detail="ok"),
        "kuzu": SimpleNamespace(ok=False, latency_ms=0.5, detail="boom"),
    }
    app.state.deep_health = dh
    return app


# ---------------------------------------------------------------------------
# /observability/summary
# ---------------------------------------------------------------------------


@pytest.fixture
def prom_stub() -> dict[str, Any]:
    return {
        'up{job="nexi-engine"}': [_sample({"job": "nexi-engine"}, 1.0)],
        "100 * DCGM_FI_DEV_FB_USED": [_sample({"gpu": "0"}, 91.666)],
        'up{job="vllm-node-b"}': [_sample({"job": "vllm-node-b"}, 1.0)],
        "DCGM_FI_DEV_FB_USED": [_sample({"gpu": "0"}, 22000.0)],
        "DCGM_FI_DEV_FB_FREE": [_sample({"gpu": "0"}, 2000.0)],
        "DCGM_FI_DEV_GPU_TEMP": [_sample({"gpu": "0"}, 71.0)],
        "DCGM_FI_DEV_GPU_UTIL": [_sample({"gpu": "0"}, 84.0)],
        'name="vllm-ornith.service"': [_sample({}, 1.0)],
        '__RANGE__nothing': [],
    }


@pytest.mark.asyncio
async def test_summary_surfaces_nodes_gpu_lock_and_tiers(
    prom_stub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_app(prom_stub, monkeypatch)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/observability/summary")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["nodes"]["b"]["nexi_up"] is True
    assert body["nodes"]["b"]["vllm_up"] is True
    assert body["gpu"]["vram_used_pct"] == pytest.approx(91.666, abs=0.05)
    assert body["gpu"]["temp_c"] == 71.0
    assert body["lock_holder"] == "ornith"
    tiers = body["memory_tiers"]
    assert tiers["kuzu"]["ok"] is False
    assert tiers["redis"]["ok"] is True
    assert "hitl" in body and "alerts_firing" in body


@pytest.mark.asyncio
async def test_lock_holder_derivation() -> None:
    from xnch.routes.observability import derive_lock_holder

    assert derive_lock_holder(True, False) == "ornith"
    assert derive_lock_holder(False, True) == "vision_stack"
    assert derive_lock_holder(False, False) == "none"
    assert derive_lock_holder(None, None) == "unknown"
    # Conflicts= makes both-active impossible; if observed anyway, flag it.
    assert derive_lock_holder(True, True) == "contention"


@pytest.mark.asyncio
async def test_summary_degrades_gracefully_when_prometheus_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = {'up{job="nexi-engine"}': None}  # 503 from stub => unavailable
    app = _build_app(stub, monkeypatch)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/observability/summary")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert "memory_tiers" in body  # local signals still present


# ---------------------------------------------------------------------------
# /observability/hitl
# ---------------------------------------------------------------------------


@pytest.fixture
def hitl_stub(prom_stub: dict[str, Any]) -> dict[str, Any]:
    stub = dict(prom_stub)
    queue_series = [
        {
            "metric": {},
            "values": [[0, "1"], [60, "2"], [120, "0"]],
        }
    ]
    stub["__RANGE__xnch_hitl_pending_interrupts"] = queue_series
    stub['increase(xnch_hitl_decisions_total{decision="approved"}'] = [_sample({}, 7.0)]
    stub['increase(xnch_hitl_decisions_total{decision="rejected"}'] = [_sample({}, 2.0)]
    stub["increase(xnch_hitl_gate_bypass_total[24h]"] = [_sample({}, 1.0)]
    stub["xnch_hitl_time_to_decision_seconds_bucket"] = [
        {"metric": {"le": "+Inf"}, "value": [0, "4"]},
        {"metric": {"le": "60"}, "value": [0, "3"]},
        {"metric": {"le": "300"}, "value": [0, "4"]},
    ]
    return stub


@pytest.mark.asyncio
async def test_hitl_surface_returns_queue_trend_rates_and_ttd(
    hitl_stub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_app(hitl_stub, monkeypatch)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/observability/hitl")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["queue_depth_series"][0]["points"] == [[0, 1.0], [60, 2.0], [120, 0.0]]
    assert body["decisions_1h"]["approved"] == 7.0
    assert body["decisions_1h"]["rejected"] == 2.0
    assert body["bypass_24h"] == 1.0
    assert body["last_bypass_alert"]["labels"]["alertname"] == "HitlGateBypassFiring"
    buckets = {b["le"]: b["count"] for b in body["time_to_decision_buckets"]}
    assert buckets["+Inf"] == 4.0 and buckets["60"] == 3.0
    assert body["expires_1h"] is None
    assert "expiry_note" in body


@pytest.mark.asyncio
async def test_hitl_surface_window_params_clamped(
    hitl_stub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_app(hitl_stub, monkeypatch)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/observability/hitl", params={"window_s": "999999999", "step_s": "1"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /observability/inference
# ---------------------------------------------------------------------------


@pytest.fixture
def inference_stub(prom_stub: dict[str, Any]) -> dict[str, Any]:
    stub = dict(prom_stub)
    util_series = [{"metric": {"gpu": "0"}, "values": [[0, "10"], [15, "80"], [30, "45"]]}]
    vram_series = [{"metric": {"gpu": "0"}, "values": [[0, "90"], [15, "92"], [30, "93.5"]]}]
    tps_series = [{"metric": {}, "values": [[0, "31.5"], [15, "42.0"], [30, "38.2"]]}]
    queue = [
        {"metric": {"__name__": "vllm:num_requests_running"}, "values": [[0, "1"], [15, "2"]]},
        {"metric": {"__name__": "vllm:num_requests_waiting"}, "values": [[0, "0"], [15, "3"]]},
    ]
    stub["__RANGE__DCGM_FI_DEV_GPU_UTIL"] = util_series
    stub["__RANGE__100 * DCGM_FI_DEV_FB_USED"] = vram_series
    stub["__RANGE__generation_tokens_total"] = tps_series
    stub["__RANGE__num_requests_"] = queue
    stub["histogram_quantile(0.5,"] = [_sample({}, 1.25)]
    stub["histogram_quantile(0.95,"] = [_sample({}, 4.75)]
    return stub


@pytest.mark.asyncio
async def test_inference_surface_returns_throughput_latency_queue(
    inference_stub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_app(inference_stub, monkeypatch)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/observability/inference")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["gpu_util_series"][0]["points"][-1] == [30, 45.0]
    assert body["vram_pct_series"][0]["points"][-1] == [30, 93.5]
    assert body["tokens_per_sec_series"][0]["points"][-1] == [30, 38.2]
    qd = {s["metric"].get("__name__"): s["points"] for s in body["queue_depth_series"]}
    assert "vllm:num_requests_running" in qd and "vllm:num_requests_waiting" in qd
    assert body["latency_p50_s"] == 1.25
    assert body["latency_p95_s"] == 4.75


@pytest.mark.asyncio
async def test_inference_surface_handles_missing_vllm_metrics(
    prom_stub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_app(prom_stub, monkeypatch)  # no __RANGE__ routes at all
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/observability/inference")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["tokens_per_sec_series"] == []
    assert body["latency_p50_s"] is None
