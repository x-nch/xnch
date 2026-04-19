"""xnch-server v0 — governance, memory, and authorization service."""
import logging
from contextlib import asynccontextmanager
from starlette.requests import Request
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from .auth import load_or_generate_keypair, GovernanceStore, TokenSigner, TokenVerifier
from .audit import EventLog, DecisionLedger
from .config import settings
from .learning import PatternExtractor, ScoreAdapter, PolicyCandidateGenerator
from .memory import init_db, EpisodicStore, PatternStore, KVCache
from .memory.db import get_state_version, get_policy_version, increment_state_version
from .policy import PolicyLoader, PolicyEngine
from .routes import (
    session_router, memory_router, policy_router,
    verdict_router, execution_router, governance_router, auth_router,
)

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

    # Learning
    s.pattern_extractor = PatternExtractor(s.episodic, s.pattern_store)
    s.score_adapter = ScoreAdapter(settings.db_path)
    s.policy_candidates = PolicyCandidateGenerator(s.pattern_store, settings.db_path)

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
    scheduler.add_job(
        lambda: __import__('asyncio').get_event_loop().create_task(
            s.pattern_extractor.run()
        ),
        "cron", hour="*/6", id="pattern_extractor",
    )
    scheduler.add_job(
        lambda: __import__('asyncio').get_event_loop().create_task(
            s.score_adapter.evaluate()
        ),
        "cron", hour="*/6", minute=30, id="score_adapter",
    )
    scheduler.add_job(
        lambda: __import__('asyncio').get_event_loop().create_task(
            s.policy_candidates.run()
        ),
        "cron", hour="*/6", minute=45, id="policy_candidates",
    )
    scheduler.start()
    s.scheduler = scheduler

    s.event_log.emit("startup", "xnch", "SERVER_STARTED", data={"version": "0.1.0"})
    logger.info("xnch-server started")

    yield

    scheduler.shutdown(wait=False)
    await s.kv_cache.aclose()
    s.event_log.emit("shutdown", "xnch", "SERVER_STOPPED")


app = FastAPI(title="xnch", version="0.1.0", lifespan=lifespan)

app.include_router(session_router)
app.include_router(memory_router)
app.include_router(policy_router)
app.include_router(verdict_router)
app.include_router(execution_router)
app.include_router(governance_router)
app.include_router(auth_router)


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
