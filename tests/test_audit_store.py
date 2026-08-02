"""Tests for the Postgres audit emitter (xnch/memory/audit_store.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xnch.memory import audit_store


class TestAuditWrite:
    """The _write coroutine persists a JSONB payload row."""

    async def test_write_inserts_row(self):
        emitter = audit_store._AuditEmitter()
        conn = MagicMock()
        conn.execute = AsyncMock()
        pool = MagicMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        pool.acquire.return_value.__aexit__.return_value = None
        emitter._pool = pool

        await emitter._write("trace-1", "intent_interpreter", "CLASSIFY_START", {"risk": 0.1})

        args = conn.execute.call_args[0]
        assert "INSERT INTO audit_events" in args[0]
        assert args[1] == "trace-1"
        assert args[2] == "intent_interpreter"
        assert args[3] == "CLASSIFY_START"
        assert '"risk": 0.1' in args[4]

    async def test_write_none_payload_uses_null(self):
        emitter = audit_store._AuditEmitter()
        conn = MagicMock()
        conn.execute = AsyncMock()
        pool = MagicMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        pool.acquire.return_value.__aexit__.return_value = None
        emitter._pool = pool

        await emitter._write(None, "comp", "EVENT", None)
        args = conn.execute.call_args[0]
        assert args[1] is None
        assert args[4] is None


class TestAuditEmit:
    """emit() is fire-and-forget and never raises."""

    def test_emit_before_pool_ready_is_noop(self):
        emitter = audit_store._AuditEmitter()
        with patch.object(emitter, "_ensure"):
            emitter.emit("trace-1", "comp", "EVENT", {})  # must not raise

    def test_emit_event_module_level_never_raises(self):
        mock_emitter = MagicMock()
        with patch.object(audit_store, "_emitter", mock_emitter):
            audit_store.emit_event("trace-1", "comp", "EVENT", {"k": 1})
        mock_emitter.emit.assert_called_once_with(
            "trace-1", "comp", "EVENT", {"k": 1}
        )

    def test_emit_event_coerces_uuid(self):
        from uuid import uuid4

        mock_emitter = MagicMock()
        with patch.object(audit_store, "_emitter", mock_emitter):
            audit_store.emit_event(uuid4(), "comp", "EVENT", None)
        emitted_trace = mock_emitter.emit.call_args[0][0]
        assert isinstance(emitted_trace, str)
