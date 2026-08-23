"""Prometheus instrumentation for xnch: HTTP, HITL gate, consolidation, memory tiers.

Exposes GET /metrics (Prometheus text format). The endpoint is scrape-gated:
only requests from hosts matching `settings.metrics_allow_cidrs` are served;
everyone else gets 403. This keeps /metrics off the public/LAN-untrusted
surface given prior findings about unauthenticated services.
"""
from __future__ import annotations

import ipaddress
import time
from typing import Any, Iterable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from ..config import settings

HTTP_REQUESTS = Counter(
    "xnch_http_requests_total",
    "HTTP requests processed, by method/route-template/status.",
    ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "xnch_http_request_seconds",
    "End-to-end request latency by route template.",
    ["method", "route"],
)

HITL_INTERRUPTS_OPENED = Counter(
    "xnch_hitl_interrupts_opened_total",
    "HITL gate interrupts opened (EXECUTION proposals awaiting a human decision).",
)
HITL_DECISIONS = Counter(
    "xnch_hitl_decisions_total",
    "Human decisions on HITL interrupts, by decision (approved/rejected).",
    ["decision"],
)
HITL_PENDING = Gauge(
    "xnch_hitl_pending_interrupts",
    "Interrupts currently awaiting a human decision.",
)
HITL_OLDEST_PENDING_AGE = Gauge(
    "xnch_hitl_oldest_pending_age_seconds",
    "Age in seconds of the oldest pending interrupt; 0 when queue is empty.",
)
HITL_GATE_BYPASS = Counter(
    "xnch_hitl_gate_bypass_total",
    "EXECUTION decisions allowed outside the HITL interrupt gate, by origin "
    "(goal_loop today; the alert on this series is security-critical).",
    ["origin"],
)
HITL_TIME_TO_DECISION = Histogram(
    "xnch_hitl_time_to_decision_seconds",
    "Time from interrupt opened to human decision (approve/reject).",
    ["decision"],
)

CONSOLIDATION_RUNS = Counter(
    "xnch_consolidation_runs_total",
    "Consolidation job completions by result (success/failure).",
    ["result"],
)
CONSOLIDATION_SECONDS = Histogram(
    "xnch_consolidation_run_seconds",
    "Duration of consolidation runs.",
)

SQLITE_QUERY_SECONDS = Histogram(
    "xnch_sqlite_query_seconds",
    "aiosqlite query latency for instrumented memory-store operations.",
    ["store"],
)

MEMORY_TIER_UP = Gauge(
    "xnch_memory_tier_up",
    "Deep health probe result per memory tier (1=probe passed, 0=failed/unavailable).",
    ["tier"],
)
MEMORY_TIER_PROBE_SECONDS = Gauge(
    "xnch_memory_tier_probe_seconds",
    "Latency of the last deep-health round-trip per tier.",
    ["tier"],
)
MEMORY_TIER_LAST_SUCCESS = Gauge(
    "xnch_memory_tier_last_success_unixtime",
    "Unix timestamp of the last successful deep probe per tier.",
    ["tier"],
)

_DECISION_LABELS = {"approve": "approved", "reject": "rejected"}

_pending_since: dict[str, float] = {}
hitl_pending_interrupts = HITL_PENDING


def record_interrupt_opened(thread_id: str) -> None:
    """Track a new pending HITL interrupt."""
    _pending_since[thread_id] = time.time()
    HITL_INTERRUPTS_OPENED.inc()
    _refresh_pending_gauges()


def record_decision(thread_id: str, decision: str) -> None:
    """Record a human decision ('approve'/'reject' or 'approved'/'rejected')."""
    label = _DECISION_LABELS.get(decision, decision)
    if label not in ("approved", "rejected"):
        raise ValueError(f"unknown decision {decision!r}")
    opened_at = _pending_since.pop(thread_id, None)
    if opened_at is not None:
        HITL_TIME_TO_DECISION.labels(decision=label).observe(max(0.0, time.time() - opened_at))
    HITL_DECISIONS.labels(decision=label).inc()
    _refresh_pending_gauges()


def _refresh_pending_gauges() -> None:
    HITL_PENDING.set(len(_pending_since))
    oldest = min(_pending_since.values()) if _pending_since else 0.0
    HITL_OLDEST_PENDING_AGE.set(max(0.0, time.time() - oldest) if _pending_since else 0.0)


def hitl_pending_snapshot() -> dict[str, Any]:
    """JSON-able snapshot of pending interrupts for UI/health surfaces."""
    now = time.time()
    items = [
        {"thread_id": tid, "age_seconds": round(now - since, 3)}
        for tid, since in sorted(_pending_since.items(), key=lambda kv: kv[1])
    ]
    return {
        "pending_count": len(items),
        "oldest_age_seconds": items[0]["age_seconds"] if items else 0.0,
        "interrupts": items,
    }


def host_allowed(host: str | None, allowlist: Iterable[str]) -> bool:
    """True if host is an exact allowlist entry or falls inside an allowed CIDR."""
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return host in set(allowlist)
    for entry in allowlist:
        try:
            if addr in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        method = request.method
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = getattr(request.scope.get("route"), "path_format", None) or "unmatched"
            HTTP_REQUESTS.labels(method=method, route=route, status=str(status)).inc()
            HTTP_LATENCY.labels(method=method, route=route).observe(time.perf_counter() - start)


def install_metrics_middleware(app: Any) -> None:
    app.add_middleware(MetricsMiddleware)


async def metrics_endpoint(request: Request) -> Response:
    """GET /metrics — Prometheus scrape target, restricted to internal networks."""
    client_host = request.client.host if request.client else ""
    if not host_allowed(client_host, list(settings.metrics_allow_cidrs)):
        return Response("forbidden", status_code=403)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def timed_sqlite(store: str) -> Any:
    """Decorator timing an async store method into SQLITE_QUERY_SECONDS."""

    def decorator(fn: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with SQLITE_QUERY_SECONDS.labels(store=store).time():
                return await fn(*args, **kwargs)

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator
