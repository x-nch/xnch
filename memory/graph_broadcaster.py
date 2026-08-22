"""In-process pub/sub for live Kuzu graph mutations."""

from __future__ import annotations

import asyncio
from typing import Any


class GraphBroadcaster:
    """Fan-out graph mutation events to SSE subscribers."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    def publish(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._put, queue, event)
                else:
                    self._put(queue, event)
            except RuntimeError:
                pass

    @staticmethod
    def _put(queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
