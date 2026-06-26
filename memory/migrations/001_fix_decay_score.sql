-- Remove broken GENERATED ALWAYS column, replace with plain FLOAT
ALTER TABLE memory DROP COLUMN IF EXISTS decay_score;
ALTER TABLE memory ADD COLUMN decay_score FLOAT DEFAULT 1.0;
-- Consolidation job computes this on schedule, not DB engine
