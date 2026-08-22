"""Scheduled OpenCode session ingestion (incremental mode).

Scheduler-facing wrapper around ingest_sessions: picks up sessions not yet
in the ledger, project-filtered from settings. Designed to run against the
live app stores from the xnch lifespan; also usable standalone.
"""

from __future__ import annotations

import logging
from pathlib import Path

from xnch.config import settings
from xnch.memory.graph_store import GraphStore
from xnch.memory.pg_episodic_store import PgEpisodicStore
from xnch.memory.session_ingest import IngestReport, ingest_sessions

logger = logging.getLogger(__name__)


def _project_directories() -> list[str] | None:
    configured = settings.session_ingest_project_dirs
    if not configured:
        return None
    return [d.strip() for d in configured.split(",") if d.strip()]


async def run_incremental_ingest(
    pg_episodic: PgEpisodicStore | None = None,
    graph_store: GraphStore | None = None,
) -> IngestReport | None:
    """Ingest not-yet-ingested sessions; returns None if the DB is absent."""
    db_path = Path(settings.session_ingest_db_path).expanduser()
    if not db_path.exists():
        logger.info("Session ingest skipped: no OpenCode database at %s", db_path)
        return None

    own_pg = pg_episodic is None
    own_graph = graph_store is None
    pg = pg_episodic or PgEpisodicStore()
    graph = graph_store or GraphStore()
    if own_pg:
        await pg.connect()
    if own_graph:
        graph.connect()
    try:
        return await ingest_sessions(
            db_path,
            pg,
            graph,
            directories=_project_directories(),
            dry_run=False,
        )
    except Exception:
        logger.exception("Incremental session ingest failed")
        return None
    finally:
        if own_graph and graph._conn is not None:
            graph.close()
        if own_pg:
            await pg.close()
