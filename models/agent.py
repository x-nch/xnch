"""Agent dispatch wire contracts (pure Pydantic, no xnch-internal imports)."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class AgentDispatchRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=20000)]
    workspace: str | None = None


class AgentClaimRequest(BaseModel):
    runner_id: str = "runner"
    ttl_s: int = 1800


class AgentOutcomeRequest(BaseModel):
    outcome_status: Literal["DONE", "FAILED"]
    exit_code: int | None = None
    output_path: str | None = None
    error: str | None = None
    result_text: str | None = None
