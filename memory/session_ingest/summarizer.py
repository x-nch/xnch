"""LLM session summarizer routed through OpenCode Go API (DeepSeek V4).

Mirrors xnch.memory.graph_extractor._extract_litellm: the OpenCode Go API is
the primary transport. If the OpenCode Go call fails (quota, credentials,
network), we fail over to the nexi router (litellm/vLLM, then OpenRouter
free) via chat_completion_with_fallback, mirroring the chat path.
Every call is traced to Langfuse with a session-scoped trace id.
"""

from __future__ import annotations

import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Any

import litellm

from xnch.config import settings
from xnch.memory.session_ingest.models import SessionDigest, SessionSummary
from xnch.observability.langfuse_client import trace_llm_call

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """Summarize this coding-agent session.

Return ONLY a JSON object with these keys:
  - "summary": 2-4 sentence description of what was worked on and why
  - "decisions": list of durable decisions made ("chose X over Y", "changed Z")
  - "outcome": one of "success", "partial", "abandoned", or null if unclear
  - "facts": list of durable entity-relation facts worth remembering long-term,
    each as {{"subject": {{"id": str, "name": str, "type": str}},
             "relation": str,
             "object": {{"id": str, "name": str, "type": str}}}}

Session:
  title: {title}
  goal/task: {goal}
  directory: {directory}
  files touched: {files}
  tools used: {tools}

Transcript digest:
{transcript}
"""


def _model_name() -> str:
    model = settings.session_ingest_model
    return f"openai/{model}" if "/" not in model else model


async def summarize_session(digest: SessionDigest) -> SessionSummary:
    """Produce a SessionSummary for a parsed session via DeepSeek V4."""
    prompt = _SUMMARY_PROMPT.format(
        title=digest.title,
        goal=digest.goal or "(not stated)",
        directory=digest.directory,
        files=", ".join(digest.files_touched) or "(none)",
        tools=json.dumps(digest.tools_used),
        transcript=digest.transcript_digest or "(empty)",
    )
    started = time.perf_counter()
    try:
        resp, served_model, tokens_used = await _complete(prompt)
    except Exception as primary_exc:
        logger.warning(
            "Primary summarizer (%s) failed: %s; falling back to nexi router",
            _model_name(),
            primary_exc,
        )
        resp, served_model, tokens_used = await _fallback(prompt)
    latency_ms = int((time.perf_counter() - started) * 1000)
    content = str(resp.choices[0].message.content).strip()

    try:
        summary = _parse_payload(content)
    finally:
        await trace_llm_call(
            prompt=prompt[:8000],
            response=content[:4000],
            model=served_model,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            trace_id=f"session-ingest:{digest.session_id}",
        )
    logger.info("Summarized session %s", digest.session_id)
    return summary


async def _complete(prompt: str) -> tuple[Any, str, int]:
    resp = await litellm.acompletion(
        model=_model_name(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize coding-agent sessions. "
                    "Return valid JSON only, no commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        api_base=settings.opencode_go_api_url.rstrip("/"),
        api_key=settings.opencode_go_api_key or "",
        temperature=0.1,
        max_tokens=settings.session_ingest_max_tokens,
    )
    return resp, _model_name(), int(getattr(resp.usage, "total_tokens", 0) or 0)


async def _fallback(prompt: str) -> tuple[Any, str, int]:
    from nexi.adapters.llm import chat_completion_with_fallback

    body, resolution = await chat_completion_with_fallback(
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize coding-agent sessions. "
                    "Return valid JSON only, no commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        json_mode=True,
        temperature=0.1,
        max_tokens=settings.session_ingest_max_tokens,
    )
    content = str(body["choices"][0]["message"]["content"]).strip()
    return (
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))]),
        resolution.model_id,
        int((body.get("usage") or {}).get("total_tokens", 0) or 0),
    )


def _parse_payload(content: str) -> SessionSummary:
    payload = _load_json(_strip_fences(content))
    if payload == _NOT_FOUND:
        raise ValueError(f"no parseable JSON in summary output ({len(content)} chars)")
    try:
        return SessionSummary.from_llm_payload(payload)
    except Exception as exc:
        raise ValueError(f"invalid summary payload: {exc}") from exc


_NOT_FOUND = object()


def _strip_fences(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("\n", 1)
        cleaned = parts[1] if len(parts) > 1 else ""
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def _load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last > first:
            try:
                return json.loads(text[first : last + 1])
            except json.JSONDecodeError:
                pass
    return _NOT_FOUND
