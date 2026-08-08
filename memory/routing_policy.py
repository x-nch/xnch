"""Memory routing policy — episodic (pgvector) vs agentmemory (am_*)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MemoryRoutingPolicy:
    primary: str
    curated: str
    deprecate_store_note_for: frozenset[str]


def load_memory_routing_policy(path: Path) -> MemoryRoutingPolicy:
    if not path.is_file():
        repo_default = (
            Path(__file__).resolve().parents[2]
            / "infra/no-k3s/shared/memory-routing.example.yaml"
        )
        path = repo_default if repo_default.is_file() else path

    if not path.is_file():
        return MemoryRoutingPolicy(
            primary="xnch_episodic",
            curated="agentmemory",
            deprecate_store_note_for=frozenset({"nexi"}),
        )

    data = yaml.safe_load(path.read_text()) or {}
    deprecated = data.get("deprecate_store_note_for") or ["nexi"]
    return MemoryRoutingPolicy(
        primary=str(data.get("primary", "xnch_episodic")),
        curated=str(data.get("curated", "agentmemory")),
        deprecate_store_note_for=frozenset(str(a) for a in deprecated),
    )


def memory_target_for_tool(tool_name: str) -> str | None:
    if tool_name.startswith("xnch_memory_"):
        return "episodic"
    if tool_name.startswith("am_memory_"):
        return "agentmemory"
    return None
