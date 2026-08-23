"""Admin endpoints for maintenance jobs."""
import json
import logging
import time
from collections import deque
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..config import settings
from ..jobs.consolidation import run_consolidation
from ..observability.metrics import CONSOLIDATION_RUNS, CONSOLIDATION_SECONDS, host_allowed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/consolidate")
async def consolidate(request: Request) -> dict:
    """Run graph extraction and episode decay using the live app stores."""
    app = request.app.state
    start = time.perf_counter()
    try:
        counts = await run_consolidation(
            pg_episodic=app.pg_episodic,
            relationship_store=app.relationship_store,
            graph_store=app.graph_store,
        )
    except Exception:
        CONSOLIDATION_SECONDS.observe(time.perf_counter() - start)
        CONSOLIDATION_RUNS.labels(result="failure").inc()
        raise
    CONSOLIDATION_SECONDS.observe(time.perf_counter() - start)
    CONSOLIDATION_RUNS.labels(result="success").inc()
    failures = counts.get("extraction_failures", 0)
    if failures:
        logger.warning("Consolidation finished with %d extraction failures", failures)
    return {"status": "ok" if failures == 0 else "partial_failure", **counts}


@router.post("/reseed-identity")
async def reseed_identity(request: Request) -> dict:
    """Sync identity facts from nexi_character.yaml into pgvector."""
    from nexi.character.cold_start_seeder import sync_identity_memories
    from xnch.routes.nexi_gateway import _invalidate_system_prompt_cache

    app = request.app.state
    added = await sync_identity_memories(app.pg_episodic)
    _invalidate_system_prompt_cache(app)
    return {"status": "ok", "added": added}


# ---------------------------------------------------------------------------
# Alertmanager webhook receiver — solo-operator alert path
# ---------------------------------------------------------------------------


def _recent_alerts_buffer(request: Request) -> deque[dict[str, Any]]:
    buf: deque[dict[str, Any]] | None = getattr(request.app.state, "recent_alerts", None)
    if buf is None:
        buf = deque(maxlen=settings.recent_alerts_capacity)
        request.app.state.recent_alerts = buf
    return buf


def _normalize_alert(alert: dict[str, Any], received_at: float) -> dict[str, Any]:
    return {
        "status": str(alert.get("status", "unknown")),
        "labels": dict(alert.get("labels") or {}),
        "annotations": dict(alert.get("annotations") or {}),
        "starts_at": alert.get("startsAt"),
        "ends_at": alert.get("endsAt"),
        "fingerprint": str(alert.get("fingerprint", "")),
        "received_at": received_at,
    }


@router.post("/alerts")
async def receive_alerts(request: Request) -> dict[str, int]:
    """Receiver for Prometheus Alertmanager webhooks.

    CIDR-gated to internal networks (see XNCH_ALERT_WEBHOOK_ALLOW_CIDRS).
    Every alert is appended to the audit event log and an in-memory ring
    buffer surfaced at GET /admin/alerts/recent (consumed by the operator UI).
    """
    client_host = request.client.host if request.client else ""
    if not host_allowed(client_host, list(settings.alert_webhook_allow_cidrs)):
        raise HTTPException(status_code=403, detail="forbidden")

    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=422, detail="body is not valid JSON")
    alerts = payload.get("alerts") if isinstance(payload, dict) else None
    if not isinstance(alerts, list) or not all(isinstance(a, dict) for a in alerts):
        raise HTTPException(status_code=422, detail="'alerts' must be a list of objects")

    app = request.app.state
    received_at = time.time()
    buf = _recent_alerts_buffer(request)
    for raw in alerts:
        entry = _normalize_alert(raw, received_at)
        buf.appendleft(entry)
        level = "WARN" if entry["status"] == "firing" else "INFO"
        app.event_log.emit(
            entry["fingerprint"] or "alertmanager",
            "alertmanager",
            f"ALERT_{entry['status'].upper()}",
            level=level,
            message=str(entry["annotations"].get("summary", "")),
            data={
                "alertname": entry["labels"].get("alertname"),
                "severity": entry["labels"].get("severity"),
                "area": entry["labels"].get("area"),
                "starts_at": entry["starts_at"],
                "ends_at": entry["ends_at"],
            },
        )
    return {"received": len(alerts)}


@router.get("/alerts/recent")
async def recent_alerts(request: Request) -> dict[str, Any]:
    """Latest received alerts (newest first) — consumed by the operator UI."""
    buf = _recent_alerts_buffer(request)
    return {"alerts": list(buf)}
