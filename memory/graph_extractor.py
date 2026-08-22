"""Graph extractor — LLM-based entity/relation extraction.

Supports two backends:
- litellm: remote via the LiteLLM proxy (default — async network I/O)
- llama_cpp: in-process llama.cpp (opt-in via
  XNCH_GRAPH_EXTRACTOR_MODEL=llama_cpp/<file>.gguf; runs off the event loop)

The llama.cpp backend is explicit opt-in only — GGUF presence no longer
overrides the configured remote model, which previously ran CPU-bound
inference inside the API process and froze the server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import litellm

from xnch.config import settings
from xnch.memory.graph_store import GraphStore
from xnch.memory.pg_episodic_store import PgEpisodicStore
from xnch.memory import llm_backend

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """Extract entity-relation triples from the following decision episode.
Return a JSON list of objects, each with:
  - "subject": {{"id": str, "name": str, "type": str}}
  - "relation": str (e.g. "deployed_to", "triggered_by", "approved")
  - "object": {{"id": str, "name": str, "type": str}}

Episode:
{raw_text}
"""


def _use_llama_cpp() -> bool:
    """Use the in-process llama.cpp backend only when explicitly configured.

    Auto-detection by GGUF file presence is intentionally gone: it silently
    overrode the configured remote model and ran CPU-bound inference inside
    the API process. Opt in explicitly with:
      XNCH_GRAPH_EXTRACTOR_MODEL=llama_cpp/<filename>.gguf
    """
    return settings.graph_extractor_model.startswith("llama_cpp/")


async def extract_and_store(
    pg_episodic=None,
    relationship_store=None,
    graph_store: GraphStore | None = None,
) -> dict[str, int]:
    """Returns {"triples_written", "episodes_processed", "extraction_failures"}.

    Per-episode failures are skipped and retried next run (not fatal)."""
    own_store = pg_episodic is None
    if pg_episodic is None:
        pg_episodic = PgEpisodicStore()
        await pg_episodic.connect()
    own_graph = graph_store is None
    try:
        episodes = await pg_episodic.fetch_unextracted_for_graph(limit=100)

        if not episodes:
            logger.info("No unextracted episodes to process")
            return {
                "triples_written": 0,
                "episodes_processed": 0,
                "extraction_failures": 0,
            }

        graph = graph_store or GraphStore(relationship_store=relationship_store)
        if own_graph:
            graph.connect()
        try:
            triples_written = 0
            extraction_failures = 0
            processed_ids: list[str] = []
            for ep in episodes:
                raw = ep.get("raw_text") or ep.get("summary") or ""
                if not raw:
                    processed_ids.append(ep["id"])
                    continue
                try:
                    triples = await _extract_triples(raw)
                except Exception:
                    logger.warning(
                        "Skipping episode %s: extraction failed, will retry next run",
                        ep["id"],
                    )
                    extraction_failures += 1
                    continue
                for t in triples:
                    t = _normalize_triple(t)
                    if not t:
                        continue
                    graph.upsert_entity(
                        id=t["subject"]["id"],
                        name=t["subject"]["name"],
                        type_=t["subject"]["type"],
                    )
                    graph.upsert_entity(
                        id=t["object"]["id"],
                        name=t["object"]["name"],
                        type_=t["object"]["type"],
                    )
                    await graph.upsert_relation(
                        from_id=t["subject"]["id"],
                        to_id=t["object"]["id"],
                        rel_type=t["relation"],
                        confidence=0.8,
                    )
                    triples_written += 1
                processed_ids.append(ep["id"])
            if processed_ids:
                await pg_episodic.mark_graph_extracted(processed_ids)
            logger.info(
                "Wrote %d triples from %d episodes (%d failed)",
                triples_written,
                len(episodes),
                extraction_failures,
            )
            return {
                "triples_written": triples_written,
                "episodes_processed": len(processed_ids),
                "extraction_failures": extraction_failures,
            }
        finally:
            if own_graph:
                graph.close()
    finally:
        if own_store:
            await pg_episodic.close()


async def _extract_triples(text: str) -> list[dict[str, Any]]:
    if _use_llama_cpp():
        result = await _extract_llama_cpp(text)
        if result:
            return result
        logger.warning("llama.cpp returned no triples, trying LiteLLM")
    return await _extract_litellm(text)


def _normalize_triple(t: Any) -> dict[str, Any] | None:
    """Coerce a triple into the canonical {subject, relation, object} shape.

    Small models sometimes emit bare strings for subject/object instead of
    {"id", "name", "type"} dicts — normalize those here. Returns None for
    malformed entries.
    """
    if not isinstance(t, dict):
        return None
    relation = t.get("relation")
    subject = _normalize_entity(t.get("subject"))
    obj = _normalize_entity(t.get("object"))
    if not relation or not subject or not obj:
        return None
    return {"subject": subject, "relation": str(relation), "object": obj}


def _normalize_entity(e: Any) -> dict[str, str] | None:
    """Coerce an entity into {"id", "name", "type"}."""
    if isinstance(e, str):
        if not e:
            return None
        return {"id": e, "name": e, "type": "entity"}
    if isinstance(e, dict):
        ident = e.get("id") or e.get("name")
        if not ident:
            return None
        return {
            "id": str(ident),
            "name": str(e.get("name") or ident),
            "type": str(e.get("type") or "entity"),
        }
    return None


def _parse_triples_json(content: str) -> list[dict[str, Any]]:
    """Parse triples from LLM output, preferring the last complete JSON array.

    Reasoning models emit a draft array mid-chain-of-thought before the final
    answer; first/last-bracket slicing splices the two together and fails with
    Extra data. Instead, scan candidate '[' positions with raw_decode() and
    keep the last valid list (dict-valued lists win over citation noise like
    "[1]").
    """
    content = content.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    last_match: list | None = None
    idx = content.find("[")
    while idx != -1:
        try:
            value, end = decoder.raw_decode(content, idx)
        except json.JSONDecodeError:
            idx = content.find("[", idx + 1)
            continue
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            last_match = value
        idx = content.find("[", max(idx + 1, end))
    if last_match is not None:
        return last_match
    logger.warning(
        "LLM returned no parseable JSON array (%d chars); treating as no triples",
        len(content),
    )
    return []


async def _extract_litellm(text: str) -> list[dict[str, Any]]:
    """Extract triples via the LiteLLM proxy's OpenAI-compatible endpoint.

    The proxy's model ids are authoritative, so the configured id is sent
    verbatim; the litellm SDK is bypassed because its client-side router
    rejects bare ids (LLM Provider NOT provided).
    """
    import os

    from xnch.config import settings as xnch_settings

    api_key = os.environ.get("LITELLM_MASTER_KEY", "")
    api_base = xnch_settings.litellm_proxy_url.rstrip("/")
    model = xnch_settings.graph_extractor_model
    provider_hint = xnch_settings.graph_extractor_provider_hint
    if provider_hint:
        model = f"{provider_hint}/{model}"
    if len(text) > 6000:
        text = text[:6000] + "\n...[truncated]"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{api_base}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an entity-relation extractor. Return valid JSON only.",
                        },
                        {"role": "user", "content": _EXTRACTION_PROMPT.format(raw_text=text)},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
                headers=headers,
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"LiteLLM proxy returned {resp.status_code}: {resp.text[:200]}"
            )
    except Exception:
        logger.exception("LiteLLM extraction failed for episode text")
        raise
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return _parse_triples_json(content)


async def _extract_ollama(text: str) -> list[dict[str, Any]]:
    """Extract triples using Ollama via litellm."""
    try:
        from xnch.config import settings as xnch_settings
        model = getattr(xnch_settings, "graph_extractor_model", "ollama/phi3:mini")
        resp = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACTION_PROMPT.format(raw_text=text)},
                {"role": "user", "content": "Return the JSON list of triples."},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)
    except Exception:
        logger.exception("LLM extraction failed for episode text")
        raise


async def _extract_llama_cpp(text: str) -> list[dict[str, Any]]:
    """Extract triples using in-process llama-cpp-python.

    The underlying inference is synchronous and CPU-bound, so it is executed
    via asyncio.to_thread to keep it off the event loop.
    """
    try:
        messages = [
            {"role": "system", "content": _EXTRACTION_PROMPT.format(raw_text=text)},
            {"role": "user", "content": "Return the JSON list of triples."},
        ]
        content = await asyncio.to_thread(
            llm_backend.chat_completion,
            messages,
            temperature=0.1,
            max_tokens=1024,
        )
        content = llm_backend._clean_json(content)
        return json.loads(content)
    except Exception:
        logger.exception("LLM extraction failed for episode text")
        raise