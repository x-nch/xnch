"""Optional agentmemory lesson prefetch for Nexi chat context."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MAX_LESSON_CHARS = 250
_DEFAULT_LIMIT = 2


async def prefetch_agent_lessons(
    app: Any,
    query: str,
    *,
    limit: int = _DEFAULT_LIMIT,
) -> list[str]:
    """Recall top lessons via am_memory_lesson_recall (bridge). Fail-open."""
    q = query.strip()
    if not q:
        return []

    from xnch_mcp.context import ActorContext
    from xnch_mcp.registry import invoke_tool

    actor = ActorContext(actor_role="nexi", trace_id="am-prefetch", session_id="prefetch")
    try:
        result = await invoke_tool(
            app,
            actor,
            "am_memory_lesson_recall",
            {"query": q, "limit": limit},
            event_log=None,
        )
    except Exception:
        logger.debug("agentmemory lesson prefetch skipped", exc_info=True)
        return []

    lines: list[str] = []
    lessons = result.get("lessons") if isinstance(result, dict) else None
    if not lessons:
        return []

    for item in lessons[:limit]:
        if not isinstance(item, dict):
            continue
        text = (item.get("content") or item.get("text") or "").strip()
        if text:
            lines.append(text[:_MAX_LESSON_CHARS])
    return lines
