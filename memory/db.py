"""SQLite setup: WAL mode, schema migrations."""
from pathlib import Path

import aiosqlite


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS actors (
    actor_id        TEXT PRIMARY KEY,
    role            TEXT NOT NULL,
    capability_set  TEXT NOT NULL,  -- JSON array
    created_at      REAL NOT NULL DEFAULT (unixepoch()),
    updated_at      REAL NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id              TEXT PRIMARY KEY,
    decision_id             TEXT NOT NULL,
    intent_class            TEXT NOT NULL,
    action_type             TEXT NOT NULL,
    entity_class            TEXT NOT NULL,
    actor_role              TEXT NOT NULL,
    outcome                 TEXT,
    prediction_delta        REAL,
    early_reextraction_flag INTEGER,
    context_snapshot        TEXT,   -- JSON
    generation_path         TEXT DEFAULT 'MODEL',
    created_at              REAL NOT NULL DEFAULT (unixepoch()),
    completed_at            REAL,
    schema_version          TEXT DEFAULT 'ep-v1'
);

CREATE INDEX IF NOT EXISTS idx_episodes_tuple
    ON episodes(intent_class, action_type, entity_class, actor_role);
CREATE INDEX IF NOT EXISTS idx_episodes_decision ON episodes(decision_id);

CREATE TABLE IF NOT EXISTS patterns (
    pattern_id          TEXT PRIMARY KEY,
    context_signature   TEXT NOT NULL UNIQUE,
    intent_class        TEXT NOT NULL,
    action_type         TEXT NOT NULL,
    entity_class        TEXT NOT NULL,
    actor_role          TEXT NOT NULL,
    success_rate        REAL NOT NULL,
    confidence          REAL NOT NULL,
    observation_count   INTEGER NOT NULL,
    avg_prediction_delta REAL,
    extraction_run_id   TEXT,
    created_at          REAL NOT NULL DEFAULT (unixepoch()),
    updated_at          REAL NOT NULL DEFAULT (unixepoch()),
    schema_version      TEXT DEFAULT 'pt-v1'
);

CREATE TABLE IF NOT EXISTS weight_configs (
    version         TEXT PRIMARY KEY,
    intent_class    TEXT NOT NULL,
    description     TEXT,
    weights         TEXT NOT NULL,  -- JSON
    approved_at     REAL,
    approved_by     TEXT,
    is_active       INTEGER NOT NULL DEFAULT 0,
    schema_version  TEXT DEFAULT 'wc-v1'
);

CREATE TABLE IF NOT EXISTS pending_weight_configs (
    version         TEXT PRIMARY KEY,
    intent_class    TEXT NOT NULL,
    weights         TEXT NOT NULL,  -- JSON
    episode_batch   TEXT,
    proposed_at     REAL NOT NULL DEFAULT (unixepoch()),
    proposed_by     TEXT
);

CREATE TABLE IF NOT EXISTS policy_candidates (
    candidate_id        TEXT PRIMARY KEY,
    pattern_id          TEXT NOT NULL,
    rule_yaml           TEXT NOT NULL,
    triggering_pattern  TEXT NOT NULL,  -- JSON
    status              TEXT NOT NULL DEFAULT 'PENDING',
    created_at          REAL NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS experiences (
    experience_id       TEXT PRIMARY KEY,
    context_signature   TEXT NOT NULL UNIQUE,
    intent_class        TEXT NOT NULL,
    action_type         TEXT NOT NULL,
    entity_class        TEXT NOT NULL,
    actor_role          TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    lesson              TEXT NOT NULL,
    insight             TEXT NOT NULL,
    verdict             TEXT NOT NULL,
    applicability       TEXT NOT NULL,
    confidence          REAL NOT NULL,
    observation_count   INTEGER NOT NULL DEFAULT 1,
    created_at          REAL NOT NULL DEFAULT (unixepoch()),
    updated_at          REAL NOT NULL DEFAULT (unixepoch()),
    schema_version      TEXT DEFAULT 'exp-v1'
);

CREATE INDEX IF NOT EXISTS idx_experiences_tuple
    ON experiences(intent_class, entity_class, actor_role);

CREATE TABLE IF NOT EXISTS goals (
    goal_id             TEXT PRIMARY KEY,
    owner_actor_id      TEXT NOT NULL,
    objective           TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'PENDING',
    progress            TEXT NOT NULL DEFAULT '',
    steps_completed     INTEGER NOT NULL DEFAULT 0,
    max_steps           INTEGER NOT NULL DEFAULT 10,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    failure_threshold   INTEGER NOT NULL DEFAULT 3,
    last_step_outcome   TEXT,
    next_due_at         REAL,
    lease_owner         TEXT,
    lease_expires_at    REAL,
    simulation_plan     TEXT,
    created_at          REAL NOT NULL DEFAULT (unixepoch()),
    updated_at          REAL NOT NULL DEFAULT (unixepoch()),
    schema_version      TEXT DEFAULT 'goal-v1'
);

CREATE INDEX IF NOT EXISTS idx_goals_due ON goals(status, next_due_at);

-- Workflows: durable playbook definitions.
CREATE TABLE IF NOT EXISTS workflows (
    id              TEXT PRIMARY KEY,
    owner_actor_id  TEXT NOT NULL,
    name            TEXT NOT NULL,
    description     TEXT,
    trigger_json    TEXT NOT NULL DEFAULT '{}',
    steps_json      TEXT NOT NULL DEFAULT '[]',
    created_at      REAL NOT NULL DEFAULT (unixepoch()),
    updated_at      REAL NOT NULL DEFAULT (unixepoch())
);

-- Workflow runs: execution instances; steps embedded as JSON in v1
-- (promoted to a row-level table when the nexi executor lands).
CREATE TABLE IF NOT EXISTS workflow_runs (
    id               TEXT PRIMARY KEY,
    workflow_id      TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    status           TEXT NOT NULL CHECK (status IN ('RUNNING','COMPLETED','FAILED','CANCELLED')),
    trigger_json     TEXT NOT NULL DEFAULT '{}',
    steps_json       TEXT NOT NULL DEFAULT '[]',
    idempotency_key  TEXT UNIQUE,
    created_at       REAL NOT NULL DEFAULT (unixepoch()),
    updated_at       REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON workflow_runs(status, created_at);

-- Run steps: row-level runtime state for the nexi executor (P2).
-- steps_json above remains a denormalized read-model snapshot.
CREATE TABLE IF NOT EXISTS workflow_run_steps (
    step_uuid        TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    idx              INTEGER NOT NULL,
    kind             TEXT NOT NULL DEFAULT 'other',
    summary          TEXT NOT NULL DEFAULT '',
    payload_json     TEXT NOT NULL DEFAULT '{}',
    requires_approval INTEGER NOT NULL DEFAULT 1,
    status           TEXT NOT NULL CHECK (status IN ('PENDING','AWAITING_APPROVAL','APPROVED','CLAIMED','EXECUTING','RETRYING','DONE','REJECTED','EXPIRED','CANCELLED','FAILED')),
    approval_id      TEXT,
    retry_count      INTEGER NOT NULL DEFAULT 0,
    max_retries      INTEGER NOT NULL DEFAULT 3,
    next_retry_at    REAL,
    lease_owner      TEXT,
    lease_expires_at REAL,
    created_at       REAL NOT NULL DEFAULT (unixepoch()),
    updated_at       REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_steps_claim ON workflow_run_steps(status, next_retry_at, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_steps_run ON workflow_run_steps(run_id, idx);

-- Approvals: first-class HITL queue, producer-agnostic.
CREATE TABLE IF NOT EXISTS approvals (
    id              TEXT PRIMARY KEY,
    producer_type   TEXT NOT NULL CHECK (producer_type IN ('chat','tool_call','goal_step','workflow_step')),
    producer_id     TEXT NOT NULL,
    payload_json    TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL CHECK (status IN ('AWAITING_APPROVAL','APPROVED','REJECTED','EXPIRED','CANCELLED')),
    risk_class      TEXT NOT NULL DEFAULT 'low' CHECK (risk_class IN ('low','elevated')),
    decision_note   TEXT,
    decided_by      TEXT,
    decided_at      REAL,
    expires_at      REAL,
    idempotency_key TEXT UNIQUE,
    created_at      REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_approvals_status_exp ON approvals(status, expires_at);
CREATE INDEX IF NOT EXISTS idx_approvals_producer ON approvals(producer_type, created_at);

-- Agent dispatch runs: tasks queued for external coding-agent runners.
CREATE TABLE IF NOT EXISTS agent_runs (
    id               TEXT PRIMARY KEY,
    status           TEXT NOT NULL CHECK (status IN ('QUEUED','RUNNING','DONE','FAILED')),
    prompt           TEXT NOT NULL,
    workspace        TEXT NOT NULL,
    runner_id        TEXT,
    lease_expires_at REAL,
    exit_code        INTEGER,
    output_path      TEXT,
    error            TEXT,
    approval_id      TEXT,
    created_at       REAL NOT NULL DEFAULT (unixepoch()),
    updated_at       REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status, created_at);

-- Step events: append-only audit trail. Never UPDATEd, never DELETEd.
CREATE TABLE IF NOT EXISTS step_events (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    step_uuid      TEXT NOT NULL,
    event_type     TEXT NOT NULL,
    actor          TEXT NOT NULL,
    ts             REAL NOT NULL DEFAULT (unixepoch()),
    snapshot_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_step ON step_events(step_uuid, seq);

CREATE TABLE IF NOT EXISTS system_state (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

INSERT OR IGNORE INTO system_state (key, value) VALUES ('state_version', '1');
INSERT OR IGNORE INTO system_state (key, value) VALUES ('policy_version', 'v1.0');
"""


async def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_SCHEMA)
        # Migrations: additive columns for pre-existing installs.
        cur = await db.execute("PRAGMA table_info(agent_runs)")
        cols = {row[1] for row in await cur.fetchall()}
        if "approval_id" not in cols:
            await db.execute("ALTER TABLE agent_runs ADD COLUMN approval_id TEXT")
        await db.commit()


async def get_state_version(db_path: Path) -> str:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT value FROM system_state WHERE key = 'state_version'"
        ) as cursor:
            row = await cursor.fetchone()
    return f"v{row[0]}" if row else "v1"


async def get_policy_version(db_path: Path) -> str:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT value FROM system_state WHERE key = 'policy_version'"
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else "v1.0"


async def increment_state_version(db_path: Path) -> str:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT value FROM system_state WHERE key = 'state_version'"
        ) as cursor:
            row = await cursor.fetchone()
        current = int(row[0]) if row else 1
        new_version = current + 1
        await db.execute(
            "UPDATE system_state SET value = ? WHERE key = 'state_version'",
            (str(new_version),),
        )
        await db.commit()
    return f"v{new_version}"
