CREATE TABLE IF NOT EXISTS relationship_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_a_id TEXT NOT NULL,
    entity_b_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    strength FLOAT DEFAULT 1.0,
    evidence TEXT[],
    first_seen TIMESTAMPTZ DEFAULT now(),
    last_reinforced TIMESTAMPTZ DEFAULT now(),
    reinforcement_count INT DEFAULT 1,
    UNIQUE(entity_a_id, entity_b_id, relationship_type)
);
CREATE INDEX IF NOT EXISTS idx_relationship_memory_a ON relationship_memory(entity_a_id);
CREATE INDEX IF NOT EXISTS idx_relationship_memory_b ON relationship_memory(entity_b_id);
