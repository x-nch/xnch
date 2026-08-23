"""Governance API: weight configs, actor management, policy candidates."""
import json
import logging
import time
from typing import Any
from uuid import uuid4

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..config import settings
from ..learning.evolution.promotion_gate import evaluate_weight_candidate
from .workflows import require_gateway_access

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/governance", tags=["governance"])


# ------------------------------------------------------------------
# Weight configs
# ------------------------------------------------------------------

@router.get("/weights")
async def get_weights(intent_class: str, request: Request) -> dict[str, Any]:
    """Return the active weight config for an intent class."""
    app = request.app.state
    async with aiosqlite.connect(settings.base_dir / "xnch.db") as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT version, weights FROM weight_configs WHERE intent_class = ? AND is_active = 1",
            (intent_class,),
        ) as cursor:
            row = await cursor.fetchone()

    if not row:
        return _default_weights(intent_class)

    return {
        "version": row["version"],
        "intent_class": intent_class,
        "weights": json.loads(row["weights"]),
    }


@router.post(
    "/weights/propose",
    dependencies=[Depends(require_gateway_access)],
)
async def propose_weights(body: dict[str, Any], request: Request) -> dict[str, Any]:
    version = f"wc-proposed-{uuid4().hex[:8]}"
    async with aiosqlite.connect(settings.base_dir / "xnch.db") as db:
        await db.execute(
            """INSERT INTO pending_weight_configs
               (version, intent_class, weights, episode_batch, proposed_at, proposed_by)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (version, body["intent_class"], json.dumps(body["weights"]),
             body.get("episode_batch"), time.time(), body.get("proposed_by", "api")),
        )
        await db.commit()
    return {"version": version, "status": "pending"}


@router.post(
    "/weights/approve",
    dependencies=[Depends(require_gateway_access)],
)
async def approve_weights(
    version: str, request: Request, force: bool = False
) -> dict[str, Any]:
    app = request.app.state
    db_path = settings.base_dir / "xnch.db"

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT * FROM pending_weight_configs WHERE version = ?", (version,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Pending version not found")

        intent_class, weights_json = row[1], row[2]

        weights = json.loads(weights_json)
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            raise HTTPException(status_code=422, detail="Weights must sum to 1.0")
        if any(v < 0.05 for v in weights.values()):
            raise HTTPException(status_code=422, detail="Each weight must be >= 0.05")

        gate = await evaluate_weight_candidate(
            intent_class=intent_class,
            proposed_weights=weights,
        )
        if gate["status"] == "block" and not force:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "candidate regresses vs active config on recent episodes",
                    "eval": gate,
                    "hint": "re-submit with force=true to override",
                },
            )

        eval_bits = [f"eval={gate['status']}"]
        if "proposed_fitness" in gate:
            eval_bits.append(f"proposed={gate['proposed_fitness']:.4f}")
            eval_bits.append(f"current={gate['current_fitness']:.4f}")
            eval_bits.append(f"episodes={gate.get('episodes', '?')}")
        description = f"Approved from proposal ({', '.join(eval_bits)})"
        if force:
            description += " [FORCED]"

        await db.execute(
            "UPDATE weight_configs SET is_active = 0 WHERE intent_class = ?", (intent_class,)
        )
        await db.execute(
            """INSERT OR REPLACE INTO weight_configs
               (version, intent_class, description, weights, approved_at, approved_by, is_active)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            (version, intent_class, description, weights_json,
             time.time(), "operator"),
        )
        await db.execute("DELETE FROM pending_weight_configs WHERE version = ?", (version,))
        await db.commit()

    await app.increment_state_version()
    return {"version": version, "status": "active", "eval": gate, "description": description}


# ------------------------------------------------------------------
# Actors
# ------------------------------------------------------------------

@router.post(
    "/actors",
    dependencies=[Depends(require_gateway_access)],
)
async def upsert_actor(body: dict[str, Any], request: Request) -> dict[str, Any]:
    app = request.app.state
    await app.governance.upsert_actor(
        body["actor_id"], body["role"], body.get("capability_set", [])
    )
    await app.increment_state_version()
    return {"status": "ok", "actor_id": body["actor_id"]}


# ------------------------------------------------------------------
# Policy candidates
# ------------------------------------------------------------------

@router.get("/policy-candidates")
async def list_policy_candidates(request: Request) -> list[dict[str, Any]]:
    async with aiosqlite.connect(settings.base_dir / "xnch.db") as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM policy_candidates WHERE status = 'PENDING' ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


def _default_weights(intent_class: str) -> dict[str, Any]:
    _defaults = {
        "EXECUTION":  {"policy_score": 0.25, "outcome_score": 0.30, "risk_score": 0.35, "context_fit_score": 0.10},
        "QUERY":      {"policy_score": 0.20, "outcome_score": 0.30, "risk_score": 0.20, "context_fit_score": 0.30},
        "DECISION":   {"policy_score": 0.25, "outcome_score": 0.35, "risk_score": 0.25, "context_fit_score": 0.15},
        "ESCALATION": {"policy_score": 0.30, "outcome_score": 0.25, "risk_score": 0.30, "context_fit_score": 0.15},
    }
    return {
        "version": "default-v0",
        "intent_class": intent_class,
        "weights": _defaults.get(intent_class, _defaults["EXECUTION"]),
    }
