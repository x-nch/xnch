"""Audit events -> Postgres `audit_events` table.

Replaces the agentmemory-backed emit_event in nexi. Keeps the same
fire-and-forget semantics: emit_event() is synchronous, never raises, and
writes happen on a background event loop thread so pipeline call sites
need no changes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import asyncpg

from xnch.config import settings

logger = logging.getLogger(__name__)

_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id   BIGSERIAL PRIMARY KEY,
    trace_id   TEXT,
    component  TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload    JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_events_created ON audit_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_trace ON audit_events(trace_id);
"""

_INSERT = """
INSERT INTO audit_events (trace_id, component, event_type, payload)
VALUES ($1, $2, $3, $4::jsonb)
"""


class _AuditEmitter:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pool: asyncpg.Pool | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #

    def _ensure(self) -> None:
        with self._lock:
            if self._loop is not None:
                return
            self._loop = asyncio.new_event_loop()
            t = threading.Thread(
                target=self._run, name="audit-emitter", daemon=True
            )
            t.start()
            asyncio.run_coroutine_threadsafe(self._init_pool(), self._loop)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        assert self._loop is not None
        self._loop.run_forever()

    async def _init_pool(self) -> None:
        try:
            pool = await asyncpg.create_pool(
                settings.postgres_url, min_size=1, max_size=2
            )
            async with pool.acquire() as conn:
                await conn.execute(_AUDIT_DDL)
            self._pool = pool
        except Exception:
            logger.warning("audit emitter could not init PG pool; events dropped")
            self._pool = None

    def emit(
        self,
        trace_id: str | None,
        component: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        try:
            self._ensure()
            if self._loop is None or self._pool is None:
                return
            asyncio.run_coroutine_threadsafe(
                self._write(trace_id, component, event_type, payload), self._loop
            )
        except Exception:
            logger.warning("audit emit failed", exc_info=True)

    async def _write(
        self,
        trace_id: str | None,
        component: str,
        event_type: str,
        payload: dict[str, Any] | None,
    ) -> None:
        try:
            async with self._pool.acquire() as conn:
                import json

                await conn.execute(
                    _INSERT,
                    trace_id,
                    component,
                    event_type,
                    json.dumps(payload) if payload else None,
                )
        except Exception:
            logger.warning("audit write failed", exc_info=True)


_emitter = _AuditEmitter()


def emit_event(
    trace_id: str | Any,
    component: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget audit event emission via Postgres."""
    _emitter.emit(str(trace_id) if trace_id is not None else None, component, event_type, payload)
