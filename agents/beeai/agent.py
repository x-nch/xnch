"""beeAI agent builders — RequirementAgent with deterministic policy gates.

The orchestrator is a ``RequirementAgent``: the LLM does reasoning/tool-calling,
but tool availability and mutation approvals are *enforced* by the requirement
stack (see ``policies.py``), not suggested by the model.
"""
from __future__ import annotations

from typing import Any

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory

from .backend import build_chat_model
from .policies import build_requirements

ORCHESTRATOR_INSTRUCTIONS = """You are the XNCH decision orchestrator.

Follow this loop:
1. Classify the user's request: QUERY (retrieve info), DECISION (plan/support),
   or EXECUTION (take an action).
2. Gather context first:
   - xnch_memory_recall for past conversations/decisions
   - xnch_web_search (SearXNG) and/or DuckDuckGo / Wikipedia for external facts
   - xnch_status for live system health
   - OpenMeteoTool for weather
   - think when you need to reason before acting
3. Synthesize a concise answer. For DECISION/EXECUTION requests, state the
   proposed action and why, then hand off — do not bypass the policy gate.
4. Never attempt to mutate state (write notes / run commands) unless the
   operator has explicitly approved; the gate will enforce this.
Respond in the user's language, be direct, and cite what you retrieved."""


def build_orchestrator(
    *,
    tools: list[Any],
    llm: ChatModel | None = None,
    approve: bool = False,
    instructions: str = ORCHESTRATOR_INSTRUCTIONS,
) -> RequirementAgent:
    return RequirementAgent(
        llm=llm or build_chat_model(),
        tools=tools,
        memory=UnconstrainedMemory(),
        requirements=build_requirements(approve=approve, tools=tools),
        name="xnch-orchestrator",
        role="decision orchestrator",
        instructions=instructions,
    )
