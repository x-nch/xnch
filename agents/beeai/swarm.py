"""beeAI AgentWorkflow demo — a small swarm of policy-gated bees.

Two bees with distinct roles share the same requirement stack, so both are
subject to the deterministic policy gate. The workflow routes the prompt to
the right bee (and lets bees hand off).
"""
from __future__ import annotations

from typing import Any

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.workflows.agent import AgentWorkflow

from .backend import build_chat_model
from .policies import build_requirements


def _bee(
    name: str,
    role: str,
    instructions: str,
    tools: list[Any],
    llm: ChatModel,
    approve: bool,
) -> RequirementAgent:
    return RequirementAgent(
        llm=llm,
        tools=tools,
        memory=UnconstrainedMemory(),
        requirements=build_requirements(approve=approve, tools=tools),
        name=name,
        role=role,
        instructions=instructions,
    )


def build_swarm(
    *,
    tools: list[Any],
    llm: ChatModel | None = None,
    approve: bool = False,
) -> AgentWorkflow:
    llm = llm or build_chat_model()
    # context_bee: read-only xnch tools + all framework tools (no exec)
    read_tools = [
        t
        for t in tools
        if getattr(t, "name", None) != "xnch_exec_run"
    ]

    workflow = AgentWorkflow(name="xnch-swarm")
    workflow.add_agent(
        _bee(
            name="context_bee",
            role="context gatherer",
            instructions=(
                "Gather context using xnch_memory_recall, xnch_web_search, "
                "DuckDuckGo, Wikipedia, xnch_status, and OpenMeteoTool as needed. "
                "Use think to organize findings, summarize, then hand off to "
                "planner_bee."
            ),
            tools=read_tools,
            llm=llm,
            approve=approve,
        )
    )
    workflow.add_agent(
        _bee(
            name="planner_bee",
            role="decision planner",
            instructions=(
                "Using the context provided by context_bee, propose a concrete "
                "plan or answer. You may use think, Wikipedia, DuckDuckGo, "
                "OpenMeteoTool, and xnch tools. For mutating actions, request "
                "operator approval explicitly — the policy gate will enforce it."
            ),
            tools=tools,
            llm=llm,
            approve=approve,
        )
    )
    return workflow
