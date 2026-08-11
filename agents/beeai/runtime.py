"""beeAI runtime — binds request context, runs agents, emits audit events."""
from __future__ import annotations

import json
import time
from typing import Any

from beeai_framework.backend import AssistantMessage
from beeai_framework.workflows.agent import AgentWorkflowInput

from xnch_mcp.context import ActorContext

from .agent import build_orchestrator
from .swarm import build_swarm
from .tools import build_tools, reset_request_context, set_request_context


def _extract_text(response: Any) -> str:
    final_answer = getattr(response, "final_answer", None)
    if isinstance(final_answer, str):
        return final_answer
    last_message = getattr(response, "last_message", None)
    if last_message is not None and isinstance(getattr(last_message, "text", None), str):
        return last_message.text
    answer = getattr(getattr(response, "state", None), "answer", None)
    if isinstance(answer, AssistantMessage):
        return "".join(answer.get_texts())
    if isinstance(response, AssistantMessage):
        return "".join(response.get_texts())
    if isinstance(response, str):
        return response
    return json.dumps(response, default=str)


async def run_agent(
    message: str,
    *,
    app_state: Any,
    actor: ActorContext,
    event_log: Any | None = None,
    approve: bool = False,
    llm: Any | None = None,
    max_iterations: int = 8,
) -> dict[str, Any]:
    """Run the orchestrator RequirementAgent and return a normalized result."""
    set_request_context(app_state, actor, event_log)
    started = time.perf_counter()
    try:
        tools = build_tools(actor, app_state, event_log)
        agent = build_orchestrator(tools=tools, llm=llm, approve=approve)
        response = await agent.run(message, max_iterations=max_iterations)
        text = _extract_text(response)
        duration_ms = int((time.perf_counter() - started) * 1000)
        if event_log is not None:
            event_log.emit(
                actor.trace_id,
                "xnch.beeai",
                "AGENT_RUN",
                data={
                    "engine": "beeai",
                    "agent": "orchestrator",
                    "tool_count": len(tools),
                    "approve": approve,
                    "duration_ms": duration_ms,
                    "output": text[:500],
                },
            )
        return {"text": text, "tool_count": len(tools), "duration_ms": duration_ms}
    finally:
        reset_request_context()


async def run_swarm(
    message: str,
    *,
    app_state: Any,
    actor: ActorContext,
    event_log: Any | None = None,
    approve: bool = False,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Run the AgentWorkflow swarm and return the final handoff text."""
    set_request_context(app_state, actor, event_log)
    started = time.perf_counter()
    try:
        tools = build_tools(actor, app_state, event_log)
        workflow = build_swarm(tools=tools, llm=llm, approve=approve)
        response = await workflow.run([AgentWorkflowInput(prompt=message)])
        result = response.result
        text = _extract_text(result) if result is not None else str(response)
        duration_ms = int((time.perf_counter() - started) * 1000)
        if event_log is not None:
            event_log.emit(
                actor.trace_id,
                "xnch.beeai",
                "SWARM_RUN",
                data={
                    "engine": "beeai",
                    "agent": "swarm",
                    "tool_count": len(tools),
                    "approve": approve,
                    "duration_ms": duration_ms,
                    "output": text[:500],
                },
            )
        return {"text": text, "tool_count": len(tools), "duration_ms": duration_ms}
    finally:
        reset_request_context()


run_orchestrator = run_agent
