"""Admin endpoints for maintenance jobs."""
from fastapi import APIRouter, Request

from ..jobs.consolidation import run_consolidation

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/consolidate")
async def consolidate(request: Request) -> dict:
    """Run graph extraction and episode decay using the live app stores."""
    app = request.app.state
    await run_consolidation(
        pg_episodic=app.pg_episodic,
        relationship_store=app.relationship_store,
        graph_store=app.graph_store,
    )
    return {"status": "ok"}
