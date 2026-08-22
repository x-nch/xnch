"""HITL helpers — typed resume decisions + when-predicates for EXECUTION gates."""
from __future__ import annotations

from enum import StrEnum
from typing import Any


class ResumeDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


def normalize_resume(value: Any) -> bool:
    """Map resume payloads to the bool contract used by select().

    Accepts legacy bools and typed strings/dicts:
      True / "approve" / {"decision": "approve"} → True
      False / "reject" / {"decision": "reject"} → False
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        raw = value.get("decision", value.get("approved"))
        return normalize_resume(raw)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {ResumeDecision.APPROVE, "approved", "true", "yes", "1"}:
            return True
        if lowered in {ResumeDecision.REJECT, "rejected", "false", "no", "0"}:
            return False
    return bool(value)


def parse_resume_decision(
    *,
    decision: str | None = None,
    approved: bool | None = None,
) -> bool:
    """API helper: prefer typed decision, fall back to approved bool."""
    if decision is not None:
        return normalize_resume(decision)
    if approved is not None:
        return bool(approved)
    raise ValueError("resume requires decision or approved")


def _risk_from_evaluated(evaluated: list[dict[str, Any]] | None) -> float:
    """Best-effort risk signal from evaluated option dumps (0..1, higher = riskier)."""
    if not evaluated:
        return 0.0
    best = max(evaluated, key=lambda e: float(e.get("composite_score") or 0.0))
    # Prefer explicit risk fields when present; else invert composite as proxy.
    for key in ("risk_score", "risk"):
        if key in best and best[key] is not None:
            return float(best[key])
    scores = best.get("scores") or {}
    if isinstance(scores, dict) and scores.get("risk_score") is not None:
        return float(scores["risk_score"])
    composite = float(best.get("composite_score") or 0.0)
    return max(0.0, min(1.0, 1.0 - composite))


def should_interrupt_execution(
    *,
    intent_class: str,
    evaluated: list[dict[str, Any]] | None = None,
    mode: str = "always",
    risk_threshold: float = 0.5,
) -> bool:
    """when-predicate for EXECUTION HITL.

    modes:
      always — interrupt every EXECUTION (current default)
      risk_threshold — interrupt when estimated risk >= threshold
      never — skip interrupt (unsafe; tests / dry-runs only)
    """
    if intent_class != "EXECUTION":
        return False
    normalized = (mode or "always").strip().lower()
    if normalized == "never":
        return False
    if normalized == "risk_threshold":
        return _risk_from_evaluated(evaluated) >= risk_threshold
    return True
