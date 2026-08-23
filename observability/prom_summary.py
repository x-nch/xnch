"""Server-side Prometheus summarizer backing the /observability/* UI surfaces.

The browser never talks to Prometheus directly; xnch queries the local
Prometheus (:9090) and returns compact JSON. All queries fail soft:
PrometheusUnavailable => endpoints respond {"available": false, ...local data}.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from ..config import settings


class PrometheusUnavailable(Exception):
    pass


def _scalar(result: list[dict[str, Any]]) -> float | None:
    for sample in result:
        try:
            return float(sample["value"][1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return None


def _series(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in result:
        points = [[float(ts), float(v)] for ts, v in s.get("values", [])]
        out.append({"metric": dict(s.get("metric") or {}), "points": points})
    return out


class PrometheusClient:
    def __init__(
        self,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (base_url or settings.prometheus_url).rstrip("/")
        if http_client is not None:
            self._client = http_client
        else:
            self._client = httpx.AsyncClient(
                base_url=self._base_url, timeout=settings.prometheus_timeout_s
            )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        resp = await self._client.get(path, params=params)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            raise PrometheusUnavailable(f"prometheus returned {body.get('status')}")
        return body["data"]["result"]

    async def query(self, expr: str) -> list[dict[str, Any]]:
        """Instant vector query. Empty on no data; raises only on transport/5xx."""
        return await self._get("/api/v1/query", {"query": expr})

    async def query_range(
        self, expr: str, window_s: int, step_s: int
    ) -> list[dict[str, Any]]:
        end = time.time()
        start = end - max(window_s, 60)
        return await self._get(
            "/api/v1/query_range",
            {
                "query": expr,
                "start": f"{start:.3f}",
                "end": f"{end:.3f}",
                "step": f"{max(step_s, 10)}s",
            },
        )

    async def query_scalar(self, expr: str) -> float | None:
        return _scalar(await self.query(expr))

    async def first_scalar(self, exprs: list[str]) -> tuple[float | None, str]:
        """Try candidate exprs in order; return first non-empty value + its expr."""
        for expr in exprs:
            val = await self.query_scalar(expr)
            if val is not None:
                return val, expr
        return None, ""

    async def first_series(
        self, exprs: list[str], window_s: int, step_s: int
    ) -> tuple[list[dict[str, Any]], str]:
        for expr in exprs:
            result = await self.query_range(expr, window_s, step_s)
            if result:
                return _series(result), expr
        return [], ""
