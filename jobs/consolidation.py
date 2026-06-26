"""Consolidation job — summarization, graph extraction, decay, archival.
Uses agentmemory for all storage operations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from agentmemory import get_memories, update_memory

from xnch.memory.graph_extractor import extract_and_store
from xnch.memory.pg_episodic_store import CATEGORY as EPISODE_CATEGORY

logger = logging.getLogger(__name__)


async def run_consolidation(relationship_store=None) -> None:
    try:
        await _zep_summarize()

        triples = await extract_and_store(relationship_store=relationship_store)
        logger.info("Graph extraction: %d triples written", triples)

        all_episodes = get_memories(EPISODE_CATEGORY, n_results=5000)
        archived = _recompute_and_archive_decay(all_episodes)
        logger.info("Consolidation complete — %d episodes archived", archived)
    except Exception:
        logger.exception("Consolidation failed")


async def _zep_summarize() -> None:
    episodes = get_memories(EPISODE_CATEGORY, n_results=100)
    logger.info("Zep summarization: %d episodes in window", len(episodes))


def _recompute_and_archive_decay(all_episodes: list) -> int:
    now = datetime.now(timezone.utc)
    archived = 0
    for m in all_episodes:
        meta = dict(m["metadata"])
        try:
            ts = datetime.fromisoformat(meta.get("timestamp", now.isoformat()))
        except Exception:
            ts = now
        days = (now - ts).total_seconds() / 86400
        importance = float(meta.get("importance", 1.0))
        recall_count = int(meta.get("recall_count", 0))
        decay = importance * (2.718 ** (-0.1 * days)) * (1 + 0.1 * recall_count)
        meta["decay_score"] = str(round(decay, 4))
        if decay < 0.1 and meta.get("archived", "False") != "True":
            meta["archived"] = "True"
            archived += 1
        update_memory(EPISODE_CATEGORY, m["id"], metadata=meta)
    return archived
