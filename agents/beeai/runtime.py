"""beeAI runtime — binds request context, runs agents, emits audit events."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from beeai_framework.agents.errors import AgentError
from beeai_framework.backend import AssistantMessage
from beeai_framework.workflows.agent import AgentWorkflowInput

from xnch_mcp.context import ActorContext

from ...config import settings
from .agent import build_orchestrator
from .swarm import build_swarm
from .tools import build_tools, reset_request_context, set_request_context

logger = logging.getLogger(__name__)

# Keep beeAI from stacking GPU-bound runs on top of each other (vLLM still
# shares capacity with nexi/chat via max-num-seqs > 1).
_run_lock = asyncio.Lock()


class BeeaiTimeoutError(TimeoutError):
    """Raised when a beeAI agent/swarm run exceeds the configured timeout."""


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


async def _await_with_timeout(coro: Any, timeout_s: float) -> Any:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except TimeoutError as exc:
        raise BeeaiTimeoutError(
            f"beeAI run exceeded timeout of {timeout_s:.0f}s"
        ) from exc


async def run_agent(
    message: str,
    *,
    app_state: Any,
    actor: ActorContext,
    event_log: Any | None = None,
    approve: bool = False,
    llm: Any | None = None,
    max_iterations: int | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run the orchestrator RequirementAgent and return a normalized result.

    Raises:
        AgentError: beeAI could not resolve the task (e.g. iteration limit).
        BeeaiTimeoutError: wall-clock timeout exceeded (cancels the run).
    """
    iterations = max_iterations if max_iterations is not None else settings.beeai_max_iterations
    timeout = timeout_s if timeout_s is not None else settings.beeai_timeout_s

    async with _run_lock:
        set_request_context(app_state, actor, event_log)
        started = time.perf_counter()
        try:
            tools = build_tools(actor, app_state, event_log)
            agent = build_orchestrator(tools=tools, llm=llm, approve=approve)
            try:
                response = await _await_with_timeout(
                    agent.run(message, max_iterations=iterations),
                    timeout,
                )
            except AgentError:
                logger.warning(
                    "beeAI agent failed to resolve task (trace=%s iterations=%s)",
                    actor.trace_id,
                    iterations,
                )
                raise
            except BeeaiTimeoutError:
                logger.warning(
                    "beeAI agent timed out after %.0fs (trace=%s)",
                    timeout,
                    actor.trace_id,
                )
                raise
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
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run the AgentWorkflow swarm and return the final handoff text.

    Raises:
        AgentError: a bee in the swarm could not resolve the task.
        BeeaiTimeoutError: wall-clock timeout exceeded (cancels the run).
    """
    timeout = timeout_s if timeout_s is not None else settings.beeai_timeout_s

    async with _run_lock:
        set_request_context(app_state, actor, event_log)
        started = time.perf_counter()
        try:
            tools = build_tools(actor, app_state, event_log)
            workflow = build_swarm(tools=tools, llm=llm, approve=approve)
            try:
                response = await _await_with_timeout(
                    workflow.run([AgentWorkflowInput(prompt=message)]),
                    timeout,
                )
            except AgentError:
                logger.warning(
                    "beeAI swarm failed to resolve task (trace=%s)",
                    actor.trace_id,
                )
                raise
            except BeeaiTimeoutError:
                logger.warning(
                    "beeAI swarm timed out after %.0fs (trace=%s)",
                    timeout,
                    actor.trace_id,
                )
                raise
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
