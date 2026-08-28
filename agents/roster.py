from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ROSTER_PATH = Path(__file__).parent / "roster.yaml"

AgentStatus = Literal["online", "standby", "integration"]


class ComplexityPolicy(BaseModel):
    tokens_threshold: int = 4000
    latency_budget_s: int = 30
    price_ceiling_usd: float = 0.05


class ModelPolicy(BaseModel):
    tiers: list[str] = Field(default_factory=lambda: ["openrouter:auto"])
    complexity: ComplexityPolicy = Field(default_factory=ComplexityPolicy)
    default_tier: str = "openrouter:auto"


class AgentRosterEntry(BaseModel):
    key: str
    name: str
    role: str
    persona: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    model_policy: ModelPolicy = Field(default_factory=ModelPolicy)
    example_requests: list[str] = Field(default_factory=list)
    status: AgentStatus = "online"


@functools.lru_cache(maxsize=1)
def load_roster() -> list[AgentRosterEntry]:
    with ROSTER_PATH.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or []
    return [AgentRosterEntry(**entry) for entry in raw]


def get_agent(key: str) -> AgentRosterEntry | None:
    for entry in load_roster():
        if entry.key == key:
            return entry
    return None
