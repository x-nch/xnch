CREATE TABLE IF NOT EXISTS quarantine_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_type TEXT NOT NULL,
    raw_text TEXT,
    summary TEXT,
    importance FLOAT DEFAULT 1.0,
    quarantine_reason TEXT NOT NULL,
    quarantined_by TEXT NOT NULL,
    original_actor_role TEXT NOT NULL,
    original_trust_level TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    released_at TIMESTAMPTZ,
    released_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_quarantine_status ON quarantine_memories(released_at);
