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
