"""Append-only Event Log — fire-and-forget, never blocks the request path."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, events_path: Path) -> None:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = events_path

    def emit(
        self,
        trace_id: str,
        component: str,
        event_type: str,
        level: str = "INFO",
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "component": component,
            "event_type": event_type,
            "message": message,
            "data": data or {},
            "trace_id": trace_id,
        }
        try:
            with self._path.open("a") as fh:
                fh.write(json.dumps(event) + "\n")
        except Exception:
            pass
