"""Operator-UI observability surfaces: summarized Prometheus + local health, one JSON shape per screen."""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Request

from ..config import settings
from ..observability.metrics import hitl_pending_snapshot
from ..observability.prom_summary import PrometheusUnavailable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/observability", tags=["observability"])

_LOCK_ORNITH_EXPR = (
    'max(node_systemd_unit_state{name="vllm-ornith.service",state="active",job="node-exporter-b"})'
)
_LOCK_VISION_EXPR = (
    'max(node_systemd_unit_state{name=~".*vision.*media.*",state="active",job="node-exporter-b"})'
)

_TPS_EXPRS = [
    'sum(rate({__name__=~"vllm:(request_)?generation_tokens_total"}[1m]))',
    'sum(rate({__name__=~"vllm:(request_)?prompt_tokens_total"}[1m]))',
]
_QUEUE_EXPRS = [
    '{__name__=~"vllm:num_requests_(running|waiting)"}',
    '{__name__=~"vllm:num_requests_(running|waiting|waiting_prefix)"}',
]
def derive_lock_holder(ornith_active: bool | None, vision_active: bool | None) -> str:
    """Map the two systemd unit states onto the Conflicts= lock holder.

    Both-active is impossible under Conflicts=; if ever observed it means the
    exclusivity guarantee is broken, so surface it as explicit contention.
    """
    if ornith_active and vision_active:
        return "contention"
    if ornith_active:
        return "ornith"
    if vision_active:
        return "vision_stack"
    if ornith_active is None or vision_active is None:
        return "unknown"
    return "none"


def _bool_of(result: list[dict[str, Any]]) -> bool | None:
    for sample in result:
        try:
            return float(sample["value"][1]) > 0.0
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return None


async def _collect(request: Request, exprs: list[tuple[str, str]]) -> dict[str, Any]:
    """Run instant queries in parallel; on any failure mark prometheus unavailable."""
    client = getattr(request.app.state, "prom_client", None)
    out: dict[str, Any] = {}
    if client is None:
        raise PrometheusUnavailable("prometheus client not configured")

    import asyncio

    async def one(name: str, expr: str) -> None:
        result = await client.query(expr)
        scalar = None
        for sample in result:
            try:
                scalar = float(sample["value"][1])
                break
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        out[name] = scalar

    results = await asyncio.gather(*(one(n, e) for n, e in exprs), return_exceptions=True)
    for (_, _), res in zip(exprs, results):
        if isinstance(res, BaseException):
            raise PrometheusUnavailable(str(res))
    return out


@router.get("/summary")
async def summary(request: Request) -> dict[str, Any]:
    """Home "is the system healthy" screen data: nodes, GPU, lock holder, tiers, HITL."""
    app_state = request.app.state
    body: dict[str, Any] = {
        "generated_at": time.time(),
        "available": True,
        "nodes": {
            "a": {"up": True, "role": "control-plane"},
            "b": {"nexi_up": None, "vllm_up": None, "role": "gpu-inference"},
        },
        "gpu": {"vram_used_pct": None, "temp_c": None, "util_pct": None},
        "lock_holder": "unknown",
        "memory_tiers": {},
        "hitl": hitl_pending_snapshot(),
        "alerts_firing": [],
    }

    runner = getattr(app_state, "deep_health", None)
    last = getattr(runner, "last_results", None) or {}
    body["memory_tiers"] = {
        tier: {"ok": r.ok, "latency_ms": r.latency_ms, "detail": r.detail}
        for tier, r in last.items()
    }

    buf = getattr(app_state, "recent_alerts", None)
    firing = [a for a in buf or [] if a.get("status") == "firing"]
    body["alerts_firing"] = list(firing[:10])

    try:
        data = await _collect(
            request,
            [
                ("nexi_up", 'up{job="nexi-engine"}'),
                ("vllm_up", 'up{job="vllm-node-b"}'),
                ("vram_used_pct", "100 * DCGM_FI_DEV_FB_USED / clamp_min(DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE, 1)"),
                ("temp_c", "DCGM_FI_DEV_GPU_TEMP"),
                ("util_pct", "DCGM_FI_DEV_GPU_UTIL"),
                ("ornith_active", _LOCK_ORNITH_EXPR),
                ("vision_active", _LOCK_VISION_EXPR),
            ],
        )
    except PrometheusUnavailable as exc:
        logger.info("observability summary: prometheus unavailable: %s", exc)
        body["available"] = False
        return body

    body["nodes"]["b"]["nexi_up"] = bool(data.get("nexi_up"))
    body["nodes"]["b"]["vllm_up"] = bool(data.get("vllm_up"))
    body["gpu"]["vram_used_pct"] = data.get("vram_used_pct")
    body["gpu"]["temp_c"] = data.get("temp_c")
    body["gpu"]["util_pct"] = data.get("util_pct")
    body["lock_holder"] = derive_lock_holder(
        (lambda v: None if v is None else bool(v))(data.get("ornith_active")),
        (lambda v: None if v is None else bool(v))(data.get("vision_active")),
    )
    return body


@router.get("/hitl")
async def hitl_surface(
    request: Request,
    window_s: int = 6 * 3600,
    step_s: int = 60,
) -> dict[str, Any]:
    """HITL activity aggregate/trend view: queue depth over time, decision rates, TTD, bypass."""
    window = max(300, min(window_s, 7 * 24 * 3600))
    step = max(10, min(step_s, 600))
    client = getattr(request.app.state, "prom_client", None)
    if client is None:
        return {"available": False}

    body: dict[str, Any] = {
        "available": True,
        "window_s": window,
        "step_s": step,
        "queue_depth_series": [],
        "decisions_1h": {"approved": None, "rejected": None},
        "expires_1h": None,
        "expiry_note": "No interrupt expiry mechanism exists yet; expires remain 0 until the sweeper lands.",
        "time_to_decision_buckets": [],
        "pending_now": hitl_pending_snapshot(),
        "bypass_24h": None,
        "last_bypass_alert": None,
    }
    alerts_buf = getattr(request.app.state, "recent_alerts", None) or []
    for alert in alerts_buf:
        if alert.get("labels", {}).get("alertname") == "HitlGateBypassFiring":
            body["last_bypass_alert"] = alert
            break

    try:
        import asyncio

        queue_res, approved_res, rejected_res, bypass_res, bucket_res = await asyncio.gather(
            client.query_range("xnch_hitl_pending_interrupts", window, step),
            client.query('increase(xnch_hitl_decisions_total{decision="approved"}[1h])'),
            client.query('increase(xnch_hitl_decisions_total{decision="rejected"}[1h])'),
            client.query("increase(xnch_hitl_gate_bypass_total[24h])"),
            client.query("xnch_hitl_time_to_decision_seconds_bucket"),
            return_exceptions=True,
        )
        results = [queue_res, approved_res, rejected_res, bypass_res, bucket_res]
        for res in results:
            if isinstance(res, BaseException):
                raise PrometheusUnavailable(str(res))
        from ..observability.prom_summary import _scalar, _series

        body["queue_depth_series"] = _series(queue_res)
        body["decisions_1h"]["approved"] = _scalar(approved_res)
        body["decisions_1h"]["rejected"] = _scalar(rejected_res)
        body["bypass_24h"] = _scalar(bypass_res)
        buckets = [
            {"le": s["metric"].get("le", "+Inf"), "count": float(s["value"][1])}
            for s in bucket_res
        ]
        buckets.sort(key=lambda b: float(b["le"]) if b["le"] != "+Inf" else float("inf"))
        body["time_to_decision_buckets"] = buckets
    except PrometheusUnavailable as exc:
        logger.info("observability hitl: prometheus unavailable: %s", exc)
        body["available"] = False
    return body


@router.get("/inference")
async def inference_surface(
    request: Request,
    window_s: int = 3600,
    step_s: int = 15,
) -> dict[str, Any]:
    """Inference performance: GPU util/VRAM trend, Ornith throughput, queue depth, latency quantiles."""
    window = max(300, min(window_s, 24 * 3600))
    step = max(10, min(step_s, 300))
    client = getattr(request.app.state, "prom_client", None)
    if client is None:
        return {"available": False}

    body: dict[str, Any] = {
        "available": True,
        "window_s": window,
        "step_s": step,
        "gpu_util_series": [],
        "vram_pct_series": [],
        "tokens_per_sec_series": [],
        "queue_depth_series": [],
        "latency_p50_s": None,
        "latency_p95_s": None,
    }

    def _e2e_quantile(q: str) -> str:
        bases = ["vllm:e2e_request_latency_seconds", "vllm:request_latency_seconds"]
        pattern = "{__name__=~\"" + "|".join(b + "_bucket" for b in bases) + "\"}"
        return f"histogram_quantile({q}, sum by (le) (rate({pattern}[5m])))"

    p50_expr = _e2e_quantile("0.5")
    p95_expr = _e2e_quantile("0.95")

    try:
        import asyncio

        util_r, vram_r, tps_pair, queue_pair, p50_v, p95_v = await asyncio.gather(
            client.query_range("DCGM_FI_DEV_GPU_UTIL", window, step),
            client.query_range(
                "100 * DCGM_FI_DEV_FB_USED / clamp_min(DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE, 1)",
                window,
                step,
            ),
            client.first_series(_TPS_EXPRS, window, step),
            client.first_series(_QUEUE_EXPRS, window, step),
            client.first_scalar([p50_expr]),
            client.first_scalar([p95_expr]),
            return_exceptions=True,
        )
        for res in (util_r, vram_r):
            if isinstance(res, BaseException):
                raise PrometheusUnavailable(str(res))
        for pair in (tps_pair, queue_pair, p50_v, p95_v):
            val = pair[0] if isinstance(pair, tuple) else pair
            if isinstance(val, BaseException):
                raise PrometheusUnavailable(str(val))

        from ..observability.prom_summary import _series

        body["gpu_util_series"] = _series(util_r)
        body["vram_pct_series"] = _series(vram_r)
        body["tokens_per_sec_series"] = tps_pair[0] if isinstance(tps_pair, tuple) else []
        body["queue_depth_series"] = queue_pair[0] if isinstance(queue_pair, tuple) else []
        body["latency_p50_s"] = p50_v[0] if isinstance(p50_v, tuple) else None
        body["latency_p95_s"] = p95_v[0] if isinstance(p95_v, tuple) else None
    except PrometheusUnavailable as exc:
        logger.info("observability inference: prometheus unavailable: %s", exc)
        body["available"] = False
    return body
