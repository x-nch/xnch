"""Graph extractor — LLM-based entity/relation extraction using agentmemory."""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from xnch.memory.graph_store import GraphStore
from xnch.memory.pg_episodic_store import PgEpisodicStore, CATEGORY as EPISODE_CATEGORY

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """Extract entity-relation triples from the following decision episode.
Return a JSON list of objects, each with:
  - "subject": {"id": str, "name": str, "type": str}
  - "relation": str (e.g. "deployed_to", "triggered_by", "approved")
  - "object": {"id": str, "name": str, "type": str}

Episode:
{raw_text}
"""


async def extract_and_store(relationship_store=None) -> int:
    from agentmemory import get_memories

    episodes = get_memories(EPISODE_CATEGORY, n_results=100)

    if not episodes:
        logger.info("No recent episodes to extract from")
        return 0

    graph = GraphStore(relationship_store=relationship_store)
    graph.connect()
    try:
        triples_written = 0
        for ep in episodes:
            raw = ep["metadata"].get("raw_text") or ep["metadata"].get("summary") or ep.get("document") or ""
            if not raw:
                continue
            triples = await _extract_triples(raw)
            for t in triples:
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


async def _extract_triples(text: str) -> list[dict[str, Any]]:
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
