"""beeAI tools — thin wrappers over the xnch MCP tool registry.

Instead of an HTTP loopback, the wrappers call ``invoke_tool`` in-process,
which is the exact same code path as ``POST /mcp/call``: actor tier checks,
bridge lookup, and audit events. The active app state / actor / event log are
resolved from contextvars that ``xnch/agents/beeai/runtime.py`` sets before
the agent runs, so a single module-level tool set works per-request.

Only tools an actor is actually allowed to call (``list_tools_for_actor``)
are wired into the agent — the registry is the source of truth for gating.
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any

from beeai_framework.tools import tool

from xnch_mcp.context import ActorContext
from xnch_mcp.registry import invoke_tool, list_tools_for_actor

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


def build_tools(
    actor: ActorContext,
    app_state: Any | None = None,
    event_log: Any | None = None,
) -> list[Any]:
    """Return the wrapped tools the actor is allowed to call (registry-gated).

    ``app_state``/``event_log`` are optional — they are only read at call time
    from the request context, so tests can build tools with a bare actor.
    """
    allowed = {t.name for t in list_tools_for_actor(actor.actor_role)}
    return [wrapped for name, wrapped in _WRAPPED.items() if name in allowed]
