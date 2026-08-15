"""Consolidation job — graph extraction, decay, archival.

Episodes live in Postgres; graph triples land in the Kuzu store.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from xnch.memory.graph_extractor import extract_and_store
from xnch.memory.pg_episodic_store import PgEpisodicStore

logger = logging.getLogger(__name__)


async def run_consolidation(
    pg_episodic=None,
    relationship_store=None,
    graph_store=None,
) -> None:
    try:
        own_store = pg_episodic is None
        if pg_episodic is None:
            pg_episodic = PgEpisodicStore()
            await pg_episodic.connect()
        try:
            triples = await extract_and_store(
                pg_episodic=pg_episodic,
                relationship_store=relationship_store,
                graph_store=graph_store,
            )
            logger.info("Graph extraction: %d triples written", triples)

            episodes = await pg_episodic.fetch_episodes_for_decay(limit=5000)
            archived = await _recompute_and_archive_decay(pg_episodic, episodes)
            logger.info("Consolidation complete — %d episodes archived", archived)
        finally:
            if own_store:
                await pg_episodic.close()
    except Exception:
        logger.exception("Consolidation failed")


async def _recompute_and_archive_decay(
    store: PgEpisodicStore,
    episodes: list[dict],
) -> int:
    now = datetime.now(timezone.utc)
    rows: list[tuple[str, float, bool]] = []
    archived = 0
    for m in episodes:
        try:
            ts = datetime.fromisoformat(m.get("timestamp", now.isoformat()))
        except Exception:
            ts = now
        days = (now - ts).total_seconds() / 86400
        importance = float(m.get("importance", 1.0))
        recall_count = int(m.get("recall_count", 0))
        decay = importance * (2.718 ** (-0.1 * days)) * (1 + 0.1 * recall_count)
        decay_score = round(decay, 4)
        should_archive = decay < 0.1 and not bool(m.get("archived", False))
        if should_archive:
            archived += 1
        rows.append(
            (m["id"], decay_score, bool(should_archive) or bool(m.get("archived", False)))
        )
    if rows:
        await store.apply_decay_batch(rows)
    return archived
