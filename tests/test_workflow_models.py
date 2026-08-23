"""Workflow wire models — kind allowlist + server-side HITL gate re-derivation.

Loaded standalone via importlib: models/workflow.py has no xnch-internal
imports and only needs pydantic.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

XNCH_ROOT = Path(__file__).resolve().parent.parent
if str(XNCH_ROOT) not in sys.path:
    sys.path.insert(0, str(XNCH_ROOT))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, XNCH_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_wf = _load("xnch_wfm_test_models", "models/workflow.py")
WorkflowStepDef = _wf.WorkflowStepDef
RunStep = _wf.RunStep


def _step(kind: str, requires_approval: bool) -> WorkflowStepDef:
    return WorkflowStepDef(
        id="s1", kind=kind, summary="x", requires_approval=requires_approval  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("kind", ["exec_tool", "send_email"])
def test_elevated_kinds_force_gated_at_parse_time(kind: str):
    step = _step(kind, requires_approval=False)
    assert step.requires_approval is True


def test_low_risk_kind_keeps_explicit_flag():
    assert _step("write_file", requires_approval=False).requires_approval is False
    assert _step("other", requires_approval=False).requires_approval is False
    assert _step("update_memory", requires_approval=True).requires_approval is True


@pytest.mark.parametrize("kind", ["write_file", "exec_tool", "send_email",
                                  "create_goal", "update_memory", "other"])
def test_all_documented_kinds_accepted(kind: str):
    assert WorkflowStepDef(id="s", kind=kind, summary="x").kind == kind


def test_unknown_kind_rejected_not_passed_through():
    with pytest.raises(ValidationError):
        WorkflowStepDef(id="s", kind="run_arbitrary_shell", summary="x")


def test_runstep_model_enforces_gating_too():
    rs = RunStep(
        step_uuid="u", index=0, kind="exec_tool", summary="x",
        requires_approval=False,
    )
    assert rs.requires_approval is True
