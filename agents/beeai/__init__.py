"""beeAI orchestration path for xnch.

Side-by-side, feature-flagged alternative to the LangGraph decision pipeline
(`xnch/agents/pipeline_graph.py`). The existing nexi decision pipeline is
untouched — this package only adds an opt-in route (`XNCH_BEEAI_ENABLED`)
that runs a beeAI agent (RequirementAgent with deterministic policy
constraints) over the same in-process MCP tool registry that powers
`/mcp/call`.

Modules:
    backend  — ChatModel factory (OpenAI-compatible → LiteLLM proxy)
    tools    — beeAI Tool wrappers over xnch_mcp.registry.invoke_tool
    policies — deterministic policy-gate Requirements (mirrors xnch tiers)
    agent    — RequirementAgent builder (deterministic requirements)
    swarm    — AgentWorkflow demo (context + planner bees)
    runtime  — context wiring + run helpers
    route    — FastAPI router mounted only when the feature flag is on
"""
from __future__ import annotations

from .backend import StaticChatModel, build_chat_model
from .runtime import run_orchestrator, run_swarm, run_agent
from .route import beeai_router
from .tools import build_tools

__all__ = [
    "StaticChatModel",
    "beeai_router",
    "build_chat_model",
    "build_tools",
    "run_agent",
    "run_orchestrator",
    "run_swarm",
]
