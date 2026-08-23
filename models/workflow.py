"""Workflow + approval domain models (wire contracts).

Pure Pydantic — no xnch-internal imports so this module stays importable in
minimal environments (tests load it standalone via importlib).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

WorkflowTriggerKind = Literal["manual", "schedule"]
ApprovalStatus = Literal[
    "AWAITING_APPROVAL", "APPROVED", "REJECTED", "EXPIRED", "CANCELLED"
]
RunStatus = Literal["RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
StepStatus = Literal[
    "PENDING",
    "AWAITING_APPROVAL",
    "DONE",
    "REJECTED",
    "EXPIRED",
    "CANCELLED",
]
ActionKind = Literal[
    "write_file",
    "exec_tool",
    "send_email",
    "create_goal",
    "update_memory",
    "other",
]
RiskClass = Literal["low", "elevated"]

ELEVATED_KINDS: frozenset[str] = frozenset({"send_email", "exec_tool"})


class WorkflowTrigger(BaseModel):
    kind: WorkflowTriggerKind = "manual"
    cron: str | None = None


class WorkflowStepDef(BaseModel):
    id: str
    kind: ActionKind = "other"
    summary: str
    target: str | None = None
    args: dict[str, Any] | str | None = None
    preview: str | None = None
    requires_approval: bool = True
    description: str | None = None

    @model_validator(mode="after")
    def _enforce_elevated_gating(self) -> WorkflowStepDef:
        """Client never decides gating for elevated kinds (exec_tool,
        send_email). The flag is forced True at the API boundary so a
        browser-proposed workflow cannot route around HITL."""
        if self.kind in ELEVATED_KINDS:
            self.requires_approval = True
        return self


class WorkflowCreateRequest(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=200)]
    description: str | None = None
    trigger: WorkflowTrigger = Field(default_factory=WorkflowTrigger)
    steps: Annotated[list[WorkflowStepDef], Field(min_length=1)]
    owner_actor_id: str = "operator"


class WorkflowUpdateRequest(BaseModel):
    name: Annotated[str | None, Field(min_length=1, max_length=200)] = None
    description: str | None = None
    trigger: WorkflowTrigger | None = None
    steps: list[WorkflowStepDef] | None = None


class RunStep(BaseModel):
    """Runtime step embedded in workflow_runs.steps_json (v1)."""

    step_uuid: str
    index: int
    kind: ActionKind
    summary: str
    target: str | None = None
    args: dict[str, Any] | str | None = None
    preview: str | None = None
    requires_approval: bool = True
    status: StepStatus = "PENDING"
    approval_id: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: float | None = None

    @model_validator(mode="after")
    def _enforce_elevated_gating(self) -> RunStep:
        if self.kind in ELEVATED_KINDS:
            self.requires_approval = True
        return self


class ApprovalDecideRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = None
