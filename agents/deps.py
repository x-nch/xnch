"""Runtime dependencies injected into the LangGraph decision pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineDeps:
    """Stores and services wired into pipeline nodes at compile time."""

    working_memory: Any | None = None
    pg_episodic: Any | None = None
    graph_store: Any | None = None
    relationship_store: Any | None = None
    sensory_buffer: Any | None = None
    policy_engine: Any | None = None
    proactivity_engine: Any | None = None
