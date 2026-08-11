"""Deterministic policy constraints for beeAI agents.

The xnch philosophy is "deterministic rules, not LLM suggestions". These
requirements mirror that: instead of asking the model to behave well, the
framework *enforces* which tools an agent may touch and which mutations need
explicit operator approval — independent of the underlying LLM's judgement.

- ``PolicyGateRequirement``: BLOCK/allow per tool, decided by a pluggable
  checker. Blocked tools are hidden from the agent entirely (``hidden=True``).
- ``approval_requirement``: requires explicit approval before any mutating
  tool (``xnch_memory_store_note``, ``xnch_exec_run``) may run. Without the
  ``X-BeeAI-Approval: allow`` header the tools are denied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from beeai_framework.agents.requirement import RequirementAgentRunState
from beeai_framework.agents.requirement.requirements.ask_permission import (
    AskPermissionRequirement,
)
from beeai_framework.agents.requirement.requirements.requirement import (
    Requirement,
    Rule,
    run_with_context,
)
from beeai_framework.context import RunContext

from .tools import MUTATING_TOOLS

PolicyChecker = Callable[[str], "PolicyDecision"]


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str | None = None


def default_policy_checker() -> PolicyChecker:
    """Default gate: read-only tools allowed; mutating tools blocked at gate
    level (approval requirement provides the per-run allow path)."""

    def _check(tool_name: str) -> PolicyDecision:
        if tool_name in MUTATING_TOOLS:
            return PolicyDecision(
                allowed=False,
                reason="mutating tool requires explicit operator approval",
            )
        return PolicyDecision(allowed=True)

    return _check


class PolicyGateRequirement(Requirement[RequirementAgentRunState]):
    """Enforce allowed/blocked tool sets from the configured policy checker."""

    name = "policy_gate"

    def __init__(self, checker: PolicyChecker | None = None) -> None:
        super().__init__()
        self._checker = checker or default_policy_checker()
        self._tools: list[Any] = []

    async def init(self, *, tools: list[Any], ctx: RunContext) -> None:
        await super().init(tools=tools, ctx=ctx)
        self._tools = list(tools)

    @run_with_context
    async def run(self, state: RequirementAgentRunState, context: RunContext) -> list[Rule]:
        del state, context  # policy is evaluated statically per tool name
        rules: list[Rule] = []
        for tool in self._tools:
            decision = self._checker(tool.name)
            if decision.allowed:
                rules.append(Rule(target=tool.name, allowed=True))
            else:
                rules.append(
                    Rule(
                        target=tool.name,
                        allowed=False,
                        reason=decision.reason,
                        hidden=True,
                    )
                )
        return rules


def approval_requirement(
    approve: bool,
    tools: list[Any] | None = None,
) -> AskPermissionRequirement:
    """AskPermissionRequirement over the mutating tools.

    ``approve`` is bound by the caller from the request (e.g. an operator
    ``X-BeeAI-Approval: allow`` header). Default (no header) = deny.

    ``include`` is filtered to the tools actually present on this agent, so
    agents with a read-only subset (e.g. the swarm's context bee) still get a
    valid requirement stack.
    """
    present = {t.name for t in tools} if tools is not None else set(MUTATING_TOOLS)

    async def _handler(tool: Any, _input: dict[str, Any]) -> bool:
        return approve

    return AskPermissionRequirement(
        include=sorted(MUTATING_TOOLS & present),
        handler=_handler,
        remember_choices=True,
        hide_disallowed=True,
    )


def build_requirements(
    approve: bool,
    tools: list[Any] | None = None,
) -> list[Requirement]:
    """The full requirement stack for an agent."""
    return [
        PolicyGateRequirement(default_policy_checker()),
        approval_requirement(approve=approve, tools=tools),
    ]
