"""LangGraph decision pipeline runtime — checkpointer + invoke/resume.

Owns AsyncPostgresSaver (production) or an injected checkpointer (tests).
Hang create_pipeline here; nexi/main.py remains the production default path.
"""
from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any
from uuid import uuid4

from langgraph.types import Command

from ..config import settings
from ..observability.metrics import record_decision, record_interrupt_opened
from .hitl import parse_resume_decision
from .pipeline_graph import create_pipeline

logger = logging.getLogger(__name__)


def _interrupt_payloads(result: dict[str, Any] | Any) -> list[Any]:
    """Extract interrupt values from an ainvoke result or state snapshot."""
    if isinstance(result, dict) and "__interrupt__" in result:
        return [i.value for i in result["__interrupt__"]]
    return []


def _interrupts_from_snapshot(snapshot: Any) -> list[Any]:
    values: list[Any] = []
    for task in getattr(snapshot, "tasks", None) or ():
        for item in getattr(task, "interrupts", ()) or ():
            values.append(item.value)
    return values


class PipelineRuntime:
    """Compiled decision graph with a durable (or in-memory) checkpointer."""

    def __init__(
        self,
        checkpointer: Any | None = None,
        stores: dict[str, Any] | None = None,
    ) -> None:
        self._checkpointer = checkpointer
        self._stores = stores or {}
        self._graph: Any | None = None
        self._stack: AsyncExitStack | None = None
        self._owns_checkpointer = checkpointer is None

    @property
    def ready(self) -> bool:
        return self._graph is not None

    async def start(self, postgres_url: str | None = None) -> None:
        if self._graph is not None:
            return

        if self._checkpointer is None:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            url = postgres_url or settings.postgres_url
            self._stack = AsyncExitStack()
            await self._stack.__aenter__()
            self._checkpointer = await self._stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(url)
            )
            await self._checkpointer.setup()
            logger.info("AsyncPostgresSaver checkpointer ready")

        self._graph = create_pipeline(checkpointer=self._checkpointer, stores=self._stores)

    async def stop(self) -> None:
        self._graph = None
        if self._stack is not None:
            await self._stack.__aexit__(None, None, None)
            self._stack = None
            self._checkpointer = None

    def _require_graph(self) -> Any:
        if self._graph is None:
            raise RuntimeError("PipelineRuntime not started")
        return self._graph

    async def invoke(
        self,
        *,
        raw_input: str,
        session_id: str,
        trace_id: str | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        graph = self._require_graph()
        tid = thread_id or str(uuid4())
        config = {"configurable": {"thread_id": tid}}
        result = await graph.ainvoke(
            {
                "raw_input": raw_input,
                "session_id": session_id,
                "trace_id": trace_id or str(uuid4()),
                "events": [],
            },
            config,
        )
        interrupts = _interrupt_payloads(result)
        if interrupts:
            record_interrupt_opened(tid)
            return {
                "status": "interrupted",
                "thread_id": tid,
                "interrupts": interrupts,
                "state": {k: v for k, v in result.items() if k != "__interrupt__"},
            }
        return {
            "status": "completed",
            "thread_id": tid,
            "result": result,
        }

    async def resume(
        self,
        *,
        thread_id: str,
        approved: bool | None = None,
        decision: str | None = None,
    ) -> dict[str, Any]:
        graph = self._require_graph()
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config)
        pending = _interrupts_from_snapshot(snapshot)
        if not pending and not snapshot.next:
            raise LookupError(f"No pending interrupt for thread_id={thread_id}")

        approved_bool = parse_resume_decision(decision=decision, approved=approved)
        resume_payload: Any = (
            {"decision": "approve" if approved_bool else "reject"}
            if decision is not None
            else approved_bool
        )
        result = await graph.ainvoke(Command(resume=resume_payload), config)
        record_decision(thread_id, "approve" if approved_bool else "reject")
        interrupts = _interrupt_payloads(result)
        if interrupts:
            return {
                "status": "interrupted",
                "thread_id": thread_id,
                "interrupts": interrupts,
                "state": {k: v for k, v in result.items() if k != "__interrupt__"},
            }
        return {
            "status": "completed",
            "thread_id": thread_id,
            "approved": approved_bool,
            "decision": "approve" if approved_bool else "reject",
            "result": result,
        }

    async def get_pending(self, thread_id: str) -> dict[str, Any]:
        graph = self._require_graph()
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config)
        return {
            "thread_id": thread_id,
            "next": list(snapshot.next or ()),
            "interrupts": _interrupts_from_snapshot(snapshot),
            "values": snapshot.values,
        }
