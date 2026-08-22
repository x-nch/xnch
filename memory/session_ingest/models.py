"""Pydantic models for parsed and summarized OpenCode sessions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FactEntity(BaseModel):
    id: str
    name: str
    type: str = "entity"


class FactTriple(BaseModel):
    subject: FactEntity
    relation: str
    object: FactEntity


class SessionDigest(BaseModel):
    session_id: str
    title: str = ""
    directory: str = ""
    project_id: str = ""
    agent: str = ""
    model: str = ""
    started_at: datetime
    ended_at: datetime | None = None
    goal: str = ""
    files_touched: list[str] = Field(default_factory=list)
    tools_used: dict[str, int] = Field(default_factory=dict)
    transcript_digest: str = ""


class SessionSummary(BaseModel):
    summary: str
    decisions: list[str] = Field(default_factory=list)
    outcome: str | None = None
    facts: list[FactTriple] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_llm_payload(cls, payload: Any) -> SessionSummary:
        if not isinstance(payload, dict):
            raise ValueError("summary payload must be a JSON object")
        return cls.model_validate(payload)
