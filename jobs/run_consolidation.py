"""CLI entrypoint for the consolidation systemd oneshot."""
from __future__ import annotations

import asyncio
import logging

from xnch.config import settings
from xnch.jobs.consolidation import run_consolidation
from xnch.memory.pg_episodic_store import PgEpisodicStore
from xnch.memory.relationship_store import RelationshipStore

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    pg = PgEpisodicStore()
    await pg.connect()
    rel = RelationshipStore(settings.postgres_url)
    await rel.connect()
    try:
        await run_consolidation(pg_episodic=pg, relationship_store=rel)
    finally:
        await rel.close()
        await pg.close()


if __name__ == "__main__":
    asyncio.run(main())
