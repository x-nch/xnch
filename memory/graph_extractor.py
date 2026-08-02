"""Graph extractor — LLM-based entity/relation extraction.

Supports two backends:
- llama_cpp: in-process llama.cpp (default, preferred on Node A)
- ollama: external Ollama daemon (fallback)

The backend is selected at runtime based on whether a GGUF model exists
in the configured models directory.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from xnch.memory.graph_store import GraphStore
from xnch.memory.pg_episodic_store import PgEpisodicStore
from xnch.memory import llm_backend

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """Extract entity-relation triples from the following decision episode.
Return a JSON list of objects, each with:
  - "subject": {"id": str, "name": str, "type": str}
  - "relation": str (e.g. "deployed_to", "triggered_by", "approved")
  - "object": {"id": str, "name": str, "type": str}

Episode:
{raw_text}
"""


def _use_llama_cpp() -> bool:
    """Prefer the in-process llama.cpp backend when a GGUF model exists.

    Ollama is only used as a fallback when no local GGUF model is present.
    """
    try:
        return llm_backend._resolve_model_path().exists()
    except FileNotFoundError:
        return False


async def extract_and_store(pg_episodic=None, relationship_store=None) -> int:
    own_store = pg_episodic is None
    if pg_episodic is None:
        pg_episodic = PgEpisodicStore()
        await pg_episodic.connect()
    try:
        episodes = await pg_episodic.retrieve_similar(top_k=100)

        if not episodes:
            logger.info("No recent episodes to extract from")
            return 0

        graph = GraphStore(relationship_store=relationship_store)
        graph.connect()
        try:
            triples_written = 0
            for ep in episodes:
                raw = ep.get("raw_text") or ep.get("summary") or ""
                if not raw:
                    continue
                triples = await _extract_triples(raw)
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
            logger.info("Wrote %d triples from %d episodes", triples_written, len(episodes))
            return triples_written
        finally:
            graph.close()
    finally:
        if own_store:
            await pg_episodic.close()


async def _extract_triples(text: str) -> list[dict[str, Any]]:
    if _use_llama_cpp():
        return await _extract_llama_cpp(text)
    else:
        return await _extract_ollama(text)


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


async def _extract_ollama(text: str) -> list[dict[str, Any]]:
    """Extract triples using Ollama via litellm."""
    try:
        from xnch.config import settings as xnch_settings
        model = getattr(xnch_settings, "graph_extractor_model", "ollama/phi3:mini")
        resp = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {"role": "user", "content": text},
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
        return []


async def _extract_llama_cpp(text: str) -> list[dict[str, Any]]:
    """Extract triples using in-process llama-cpp-python."""
    try:
        messages = [
            {"role": "system", "content": _EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ]
        content = llm_backend.chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        content = llm_backend._clean_json(content)
        return json.loads(content)
    except Exception:
        logger.exception("LLM extraction failed for episode text")
        return []