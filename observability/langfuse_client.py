import hashlib
import json
import time
from typing import Any

import httpx

from ..config import settings


class LangfuseClient:
    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ) -> None:
        self._public_key = public_key or settings.langfuse_public_key
        self._secret_key = secret_key or settings.langfuse_secret_key
        self._host = (host or settings.langfuse_host).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._host, timeout=5.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _basic_auth(self) -> str:
        import base64
        raw = f"{self._public_key}:{self._secret_key}"
        return base64.b64encode(raw.encode()).decode()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Basic {self._basic_auth()}",
            "Content-Type": "application/json",
        }

    async def trace_llm_call(
        self,
        prompt: str,
        response: str,
        model: str,
        latency_ms: int,
        tokens_used: int,
        trace_id: str = "",
    ) -> dict[str, Any] | None:
        if not self._public_key or not self._secret_key:
            return None
        generation_id = hashlib.sha256(f"{trace_id}{time.time_ns()}".encode()).hexdigest()[:16]
        body = {
            "id": generation_id,
            "traceId": trace_id or generation_id,
            "name": "llm-call",
            "model": model,
            "modelParameters": {
                "maxTokens": tokens_used,
            },
            "usage": {
                "input": len(prompt.split()),
                "output": len(response.split()),
                "total": len(prompt.split()) + len(response.split()),
            },
            "prompt": prompt,
            "completion": response,
            "latency": latency_ms,
        }
        try:
            resp = await self._client.post(
                "/api/public/ingestion",
                headers=self._headers(),
                json={"batch": [{"type": "generation-create", "body": body}]},
            )
            resp.raise_for_status()
            return body
        except Exception:
            return None


_client: LangfuseClient | None = None


def get_client() -> LangfuseClient:
    global _client
    if _client is None:
        _client = LangfuseClient()
    return _client


async def trace_llm_call(
    prompt: str,
    response: str,
    model: str,
    latency_ms: int,
    tokens_used: int,
    trace_id: str = "",
) -> dict[str, Any] | None:
    client = get_client()
    return await client.trace_llm_call(
        prompt=prompt,
        response=response,
        model=model,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        trace_id=trace_id,
    )
