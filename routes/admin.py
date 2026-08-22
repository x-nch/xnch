"""Admin endpoints for maintenance jobs."""
import logging

from fastapi import APIRouter, Request

from ..jobs.consolidation import run_consolidation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/consolidate")
async def consolidate(request: Request) -> dict:
    """Run graph extraction and episode decay using the live app stores."""
    app = request.app.state
    counts = await run_consolidation(
        pg_episodic=app.pg_episodic,
        relationship_store=app.relationship_store,
        graph_store=app.graph_store,
    )
    failures = counts.get("extraction_failures", 0)
    if failures:
        logger.warning("Consolidation finished with %d extraction failures", failures)
    return {"status": "ok" if failures == 0 else "partial_failure", **counts}


@router.post("/reseed-identity")
async def reseed_identity(request: Request) -> dict:
    """Sync identity facts from nexi_character.yaml into pgvector."""
    from nexi.character.cold_start_seeder import sync_identity_memories
    from xnch.routes.nexi_gateway import _invalidate_system_prompt_cache

    app = request.app.state
    added = await sync_identity_memories(app.pg_episodic)
    _invalidate_system_prompt_cache(app)
    return {"status": "ok", "added": added}
