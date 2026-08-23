"""Weight-promotion regression gate.

Before a human approval activates a proposed weight config, compare its
fitness against the currently active config over recent scored decision
episodes. A measured regression blocks activation unless the operator
explicitly overrides with ``force=true``. The gate skips (fail-open) when
there is no active baseline, too little episode data, or the episode store
is unavailable — the approval path must never hard-fail on infrastructure
problems; it only blocks on measured evidence of regression.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

_LOOKBACK_DAYS = 30
_MIN_EPISODES = 10


async def evaluate_weight_candidate(
    *,
    intent_class: str,
    proposed_weights: dict[str, Any],
    lookback_days: int = _LOOKBACK_DAYS,
    min_episodes: int = _MIN_EPISODES,
    fetch_fn: Any = None,
    current_weights_fn: Any = None,
) -> dict[str, Any]:
    """Compare proposed vs active weights on recent episodes for one intent class.

    Returns ``{"status": "pass" | "block" | "skipped", ...}``. On pass/block the
    dict carries ``proposed_fitness``, ``current_fitness`` and ``episodes``;
    skips carry a machine-readable ``reason``.
    """
    try:
        fetch = fetch_fn or _default_fetch
        current_fn = current_weights_fn or _default_current_weights

        episodes = await fetch(intent_class, lookback_days)
        if len(episodes) < min_episodes:
            return {
                "status": "skipped",
                "reason": f"insufficient_data ({len(episodes)} < {min_episodes})",
            }

        current = await current_fn(intent_class)
        current_weights = current.get("weights") or {}
        if not current_weights:
            return {"status": "skipped", "reason": "no_active_baseline"}

        from .fitness import compute_fitness

        proposed_fitness = compute_fitness(proposed_weights, episodes)
        current_fitness = compute_fitness(current_weights, episodes)
    except Exception as exc:  # noqa: BLE001 — fail open on infra problems
        logger.warning("promotion gate skipped (%s): %s", type(exc).__name__, exc)
        return {"status": "skipped", "reason": f"eval_unavailable:{type(exc).__name__}"}

    status = "pass" if proposed_fitness >= current_fitness else "block"
    return {
        "status": status,
        "proposed_fitness": round(proposed_fitness, 4),
        "current_fitness": round(current_fitness, 4),
        "episodes": len(episodes),
    }


# ---------------------------------------------------------------------- #
# Production defaults (overridable in tests)                              #
# ---------------------------------------------------------------------- #

async def _default_fetch(intent_class: str, lookback_days: int) -> list[dict[str, Any]]:
    from ...memory.pg_episodic_store import PgEpisodicStore

    store = PgEpisodicStore()
    await store.connect()
    try:
        since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        episodes = await store.fetch_decision_episodes_with_scores(since)
    finally:
        await store.close()
    return [e for e in episodes if e.get("intent_class") == intent_class]


async def _default_current_weights(intent_class: str) -> dict[str, Any]:
    import json

    import aiosqlite

    from ...config import settings

    async with aiosqlite.connect(settings.base_dir / "xnch.db") as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT version, weights FROM weight_configs "
            "WHERE intent_class = ? AND is_active = 1",
            (intent_class,),
        ) as cursor:
            row = await cursor.fetchone()
    if not row:
        return {}
    try:
        weights = json.loads(row["weights"])
    except (json.JSONDecodeError, TypeError):
        return {}
    return {"version": row["version"], "weights": weights}
