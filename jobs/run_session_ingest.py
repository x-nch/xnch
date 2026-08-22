"""CLI entrypoint for OpenCode session ingestion.

Backfill historical sessions (project-filtered by default) or pick up new
ones incrementally. Scheduling/cron wiring is intentionally NOT done here;
this is a manual/oneshot entrypoint pending review.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from xnch.config import settings
from xnch.memory.graph_store import GraphStore
from xnch.memory.pg_episodic_store import PgEpisodicStore
from xnch.memory.session_ingest import IngestReport, ingest_sessions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("session-ingest")


def _parse_since_ms(value: str) -> int:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _resolve_directories(all_flag: bool, override: str | None) -> list[str] | None:
    if all_flag:
        return None
    if override:
        return [d.strip() for d in override.split(",") if d.strip()]
    configured = settings.session_ingest_project_dirs
    if configured:
        return [d.strip() for d in configured.split(",") if d.strip()]
    return [str(Path.cwd())]


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_session_ingest",
        description="Ingest OpenCode session logs into the memory tiers.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backfill", action="store_true",
                      help="ingest all matching sessions not yet ingested")
    mode.add_argument("--incremental", action="store_true",
                      help="pick up sessions not yet ingested (ledger-skipping)")
    mode.add_argument("--session-id", metavar="SID",
                      help="ingest a single session by id")
    parser.add_argument("--all", action="store_true",
                        help="disable the project-directory filter")
    parser.add_argument("--since", metavar="ISO_DATE",
                        help="only sessions updated at/after this timestamp")
    parser.add_argument("--limit", type=int, default=None,
                        help="max sessions to process this run")
    parser.add_argument("--project-dirs", metavar="P1,P2",
                        help="override project directory prefixes")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse+summarize but write nothing")
    args = parser.parse_args(argv)

    db_path = Path(settings.session_ingest_db_path).expanduser()
    if not db_path.exists():
        logger.error("OpenCode database not found at %s", db_path)
        return 1

    directories = _resolve_directories(args.all, args.project_dirs)
    session_ids = [args.session_id] if args.session_id else None
    updated_since_ms = _parse_since_ms(args.since) if args.since else None

    pg = PgEpisodicStore()
    await pg.connect()
    graph: GraphStore | None = None
    if not args.dry_run:
        graph = GraphStore()
        graph.connect()
    try:
        report = await ingest_sessions(
            db_path,
            pg,
            graph,
            session_ids=session_ids,
            updated_since_ms=updated_since_ms,
            directories=directories,
            dry_run=args.dry_run,
            limit=args.limit,
        )
    finally:
        await pg.close()
        if graph is not None:
            graph.close()

    logger.info(
        "Done: %d scanned, %d ok, %d failed, %d skipped, %d facts%s",
        report.scanned, report.succeeded, report.failed,
        report.skipped, report.facts_written,
        " (dry run)" if args.dry_run else "",
    )
    for sid, err in report.errors.items():
        logger.warning("  %s: %s", sid, err)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
