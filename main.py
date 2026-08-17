"""xnch-server v0 — governance, memory, and authorization service."""
import asyncio
import logging
from contextlib import asynccontextmanager
from starlette.requests import Request
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from .auth import load_or_generate_keypair, GovernanceStore, TokenSigner, TokenVerifier
from .audit import EventLog, DecisionLedger
from .config import settings
from .learning import PatternExtractor, PolicyCandidateGenerator
from .learning.evolution import WeightEvolver, PolicyRuleEvolver
from .memory import init_db, EpisodicStore, PatternStore, KVCache, PgEpisodicStore
from .memory import SensoryBuffer, WorkingMemory, GraphStore, RelationshipStore
from .memory.experience_store import ExperienceStore
from .memory.db import get_state_version, get_policy_version, increment_state_version
from .policy import PolicyLoader, PolicyEngine
from .routes import (
    session_router, memory_router, policy_router,
    verdict_router, execution_router, governance_router, auth_router,
    nexi_gateway_router, chat_router, admin_router, voice_router,
)
from xnch_mcp.http_router import router as mcp_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = app.state

    # Ensure directories
    for d in [settings.base_dir, settings.keys_dir, settings.audit_dir,
              settings.governance_dir, settings.policies_dir, settings.weights_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # DB init
    await init_db(settings.db_path)

    # Auth
    s.keypair = load_or_generate_keypair(settings.keys_dir)
    s.token_signer = TokenSigner(s.keypair.private_pem)
    s.token_verifier = TokenVerifier(settings.auth_secret)
    s.governance = GovernanceStore(settings.db_path)
    await s.governance.bootstrap()

    # Memory
    s.episodic = EpisodicStore(settings.db_path)
    s.pattern_store = PatternStore(settings.db_path)
    s.experience_store = ExperienceStore(settings.db_path)
    s.kv_cache = KVCache(settings.redis_url)

    # Audit
    s.event_log = EventLog(settings.audit_dir / "events.jsonl")
    s.ledger = DecisionLedger(settings.audit_dir / "decisions.jsonl")

    # Policy
    _sync_policies(settings)
    loader = PolicyLoader(settings.policies_dir)
    policy_set = loader.load()
    s.policy_engine = PolicyEngine(policy_set)
    s.policy_loader = loader

    # PG episodic store (production backend)
    s.pg_episodic = PgEpisodicStore(settings.postgres_url)
    await s.pg_episodic.connect()

    # Layer 0 — Sensory buffer (Redis perception signals)
    s.sensory_buffer = SensoryBuffer(settings.redis_url)

    # Layer 1 — Working memory (Redis session context)
    s.working_memory = WorkingMemory(settings.redis_url)

    # Layer 3 — Graph store (Kuzu semantic graph)
    from xnch.memory.graph_broadcaster import GraphBroadcaster

    s.graph_broadcaster = GraphBroadcaster()
    s.graph_broadcaster.bind_loop(asyncio.get_running_loop())
    s.relationship_store = RelationshipStore(settings.postgres_url)
    await s.relationship_store.connect()
    s.graph_store = GraphStore(
        db_path=settings.db_path,
        relationship_store=s.relationship_store,
        broadcaster=s.graph_broadcaster,
    )
    s.graph_store.connect()

    from xnch_mcp.fs.service import FsReadService

    s.fs_read_service = FsReadService.from_settings(settings)

    from xnch_mcp.exec.service import ExecRunService

    s.exec_run_service = ExecRunService.from_settings(settings)

    from xnch_mcp.web.service import WebSearchService

    s.web_search_service = WebSearchService.from_settings(settings)

    from xnch_mcp.bridge import McpBridgePool, set_bridge_pool

    s.mcp_bridge = None
    if settings.mcp_bridge_enabled and settings.mcp_servers_path.is_file():
        bridge = McpBridgePool.from_path(settings.mcp_servers_path)
        await bridge.start()
        set_bridge_pool(bridge)
        s.mcp_bridge = bridge
        logger.info(
            "MCP bridge active: %d tools from %d servers",
            len(bridge.all_tools()),
            len(bridge.server_status()),
        )

    # Cold-start / sync identity memories from identity_facts.yaml
    from nexi.character.cold_start_seeder import sync_identity_memories

    await sync_identity_memories(s.pg_episodic)

    # Learning
    s.pattern_extractor = PatternExtractor(s.pg_episodic, s.pattern_store)
    s.policy_candidates = PolicyCandidateGenerator(s.pattern_store, settings.db_path)
    s.weight_evolver = _build_weight_evolver(s.pg_episodic)
    s.policy_evolver = _build_policy_evolver(s.pg_episodic)

    # Convenience helpers for routes
    async def _get_state_version() -> str:
        return await get_state_version(settings.db_path)

    async def _get_policy_version() -> str:
        return await get_policy_version(settings.db_path)

    async def _increment_state_version() -> str:
        v = await increment_state_version(settings.db_path)
        logger.info("system_state_version incremented to %s", v)
        return v

    s.get_state_version = _get_state_version
    s.get_policy_version = _get_policy_version
    s.increment_state_version = _increment_state_version

    # Scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(s.pattern_extractor.run, "cron", hour="*/6", id="pattern_extractor")
    scheduler.add_job(s.weight_evolver.run, "cron", hour="*/6", minute=30, id="weight_evolver")
    scheduler.add_job(s.policy_evolver.run, "cron", hour="*/6", minute=45, id="policy_evolver")
    scheduler.add_job(s.policy_candidates.run, "cron", hour="*/6", minute=50, id="policy_candidates")
    scheduler.start()
    s.scheduler = scheduler

    s.event_log.emit("startup", "xnch", "SERVER_STARTED", data={"version": "0.1.0"})
    logger.info("xnch-server started")

    yield

    if s.mcp_bridge is not None:
        await s.mcp_bridge.stop()
        set_bridge_pool(None)

    scheduler.shutdown(wait=False)
    await s.kv_cache.aclose()
    await s.sensory_buffer.aclose()
    await s.working_memory.aclose()
    await s.relationship_store.close()
    await s.pg_episodic.close()
    s.event_log.emit("shutdown", "xnch", "SERVER_STOPPED")


app = FastAPI(title="xnch", version="0.1.0", lifespan=lifespan)

app.include_router(session_router)
app.include_router(memory_router)
app.include_router(policy_router)
app.include_router(verdict_router)
app.include_router(execution_router)
app.include_router(governance_router)
app.include_router(auth_router)
app.include_router(nexi_gateway_router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(voice_router)
app.include_router(mcp_router)

if settings.beeai_enabled:
    from xnch.agents.beeai.route import beeai_router

    app.include_router(beeai_router)
    logger.info("beeAI orchestration router mounted (XNCH_BEEAI_ENABLED=true)")


@app.get("/health")
async def health(request: Request) -> dict:
    redis_ok = await request.app.state.kv_cache.ping()
    state_version = await request.app.state.get_state_version()
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "ok" if redis_ok else "unavailable",
        "state_version": state_version,
        "version": "0.1.0",
    }


@app.get("/system/state")
async def system_state(request: Request) -> dict:
    state_version = await request.app.state.get_state_version()
    policy_version = await request.app.state.get_policy_version()
    return {"system_state_version": state_version, "policy_version": policy_version}


def _sync_policies(s) -> None:
    """Copy bundled default policy to ~/.xnch/policies/ if not already present."""
    import pathlib, shutil
    bundled = pathlib.Path(__file__).parent.parent.parent / "policies" / "default.yaml"
    target = s.policies_dir / "default.yaml"
    if bundled.exists() and not target.exists():
        shutil.copy(bundled, target)



def _build_weight_evolver(pg_episodic) -> WeightEvolver:
    """Wire the WeightEvolver to the app's PG episodic store and governance API."""
    from datetime import datetime, timedelta, timezone

    async def fetch_since() -> list:
        since = datetime.now(timezone.utc) - timedelta(days=30)
        return await pg_episodic.fetch_decision_episodes_with_scores(since)

    return WeightEvolver(fetch_fn=fetch_since)


def _build_policy_evolver(pg_episodic) -> PolicyRuleEvolver:
    """Wire the PolicyRuleEvolver to the app's PG episodic store and policy_candidates table."""
    from datetime import datetime, timedelta, timezone

    async def fetch_episodes() -> list:
        since = datetime.now(timezone.utc) - timedelta(days=30)
        return await pg_episodic.fetch_decision_episodes_with_scores(since)

    return PolicyRuleEvolver(episodes_fn=fetch_episodes)
