"""beeAI tools — xnch MCP wrappers + beeAI Framework built-ins.

xnch tools call ``invoke_tool`` in-process (same path as ``POST /mcp/call``):
actor tier checks, bridge lookup, and audit events. Request context is bound
via contextvars in ``runtime.py``.

Framework tools from https://framework.beeai.dev/modules/tools are attached
to *both* the orchestrator and the swarm (Think, OpenMeteo, Wikipedia,
DuckDuckGo). Sandbox/FS/Shell built-ins stay out — xnch already gates those
via MCP (``xnch_exec_run``, fs tools).
"""
from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Any

from beeai_framework.tools import tool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool

from xnch_mcp.context import ActorContext
from xnch_mcp.registry import invoke_tool, list_tools_for_actor

logger = logging.getLogger(__name__)

_app_state_var: ContextVar[Any | None] = ContextVar("beeai_app_state", default=None)
_actor_var: ContextVar[ActorContext | None] = ContextVar("beeai_actor", default=None)
_event_log_var: ContextVar[Any | None] = ContextVar("beeai_event_log", default=None)


def set_request_context(
    app_state: Any,
    actor: ActorContext,
    event_log: Any | None = None,
) -> None:
    """Bind request context for the duration of an agent run."""
    _app_state_var.set(app_state)
    _actor_var.set(actor)
    _event_log_var.set(event_log)


def reset_request_context() -> None:
    """Unbind request context after an agent run."""
    _app_state_var.set(None)
    _actor_var.set(None)
    _event_log_var.set(None)


async def _invoke(name: str, arguments: dict[str, Any]) -> str:
    app_state = _app_state_var.get()
    actor = _actor_var.get()
    if app_state is None or actor is None:
        raise RuntimeError("beeAI tools require an active request context")
    result = await invoke_tool(
        app_state,
        actor,
        name,
        arguments,
        event_log=_event_log_var.get(),
    )
    return json.dumps(result, default=str)


@tool
async def xnch_memory_recall(query: str, top_k: int = 5) -> str:
    """Semantic search over xnch episodic memory (pgvector L2). Use for conversation continuity, 'what did we discuss?', past decisions."""
    return await _invoke("xnch_memory_recall", {"query": query, "top_k": top_k})


@tool
async def xnch_web_search(query: str, limit: int = 5) -> str:
    """Search the public web via self-hosted SearXNG (no commercial API). Use for current events, release notes, external docs."""
    return await _invoke("xnch_web_search", {"query": query, "limit": limit})


@tool
async def xnch_status() -> str:
    """Query xnch system status (services, memory, graph). Read-only."""
    return await _invoke("xnch_status", {})


@tool
async def xnch_memory_store_note(text: str) -> str:
    """Store a short note into xnch episodic memory (pgvector). Mutating — requires policy approval."""
    return await _invoke("xnch_memory_store_note", {"text": text})


@tool
async def xnch_exec_run(command: str, host: str = "node-a") -> str:
    """Run an allowlisted shell command on node-a or node-b (read-only ops only). Mutating — requires policy approval."""
    return await _invoke("xnch_exec_run", {"command": command, "host": host})


_WRAPPED: dict[str, Any] = {
    "xnch_memory_recall": xnch_memory_recall,
    "xnch_web_search": xnch_web_search,
    "xnch_status": xnch_status,
    "xnch_memory_store_note": xnch_memory_store_note,
    "xnch_exec_run": xnch_exec_run,
}

# Tools that need explicit human approval on top of the policy gate.
MUTATING_TOOLS = frozenset({"xnch_memory_store_note", "xnch_exec_run"})

# Framework built-in names (for docs / filtering). Instantiated per build_tools call.
FRAMEWORK_TOOL_NAMES = frozenset({"think", "OpenMeteoTool", "Wikipedia", "DuckDuckGo"})


def build_framework_tools() -> list[Any]:
    """Ready-to-use beeAI Framework tools (docs: /modules/tools).

    Always includes Think + OpenMeteo. Wikipedia / DuckDuckGo are included when
    their optional extras are installed (``beeai-framework[wikipedia,duckduckgo]``).
    """
    tools: list[Any] = [ThinkTool(), OpenMeteoTool()]

    try:
        from beeai_framework.tools.search.wikipedia import WikipediaTool

        tools.append(WikipediaTool())
    except Exception as exc:  # optional extra
        logger.warning("WikipediaTool unavailable (install beeai-framework[wikipedia]): %s", exc)

    try:
        from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool

        tools.append(DuckDuckGoSearchTool())
    except Exception as exc:  # optional extra
        logger.warning(
            "DuckDuckGoSearchTool unavailable (install beeai-framework[duckduckgo]): %s",
            exc,
        )

    return tools


def build_tools(
    actor: ActorContext,
    app_state: Any | None = None,
    event_log: Any | None = None,
) -> list[Any]:
    """Return xnch MCP tools (registry-gated) + framework tools for agent *and* swarm.

    ``app_state``/``event_log`` are optional — they are only read at call time
    from the request context, so tests can build tools with a bare actor.
    """
    del app_state, event_log  # bound via contextvars at call time
    allowed = {t.name for t in list_tools_for_actor(actor.actor_role)}
    xnch_tools = [wrapped for name, wrapped in _WRAPPED.items() if name in allowed]
    return [*xnch_tools, *build_framework_tools()]
