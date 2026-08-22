"""Session ingest — reads OpenCode session logs into the memory tiers.

Pipeline: parser (SQLite read-only) -> redactor (hard gate) -> summarizer
(LiteLLM/ornith, Langfuse-traced) -> episodic (pgvector) + semantic (Kuzu,
bi-temporal). Orchestration lives in ingestor.ingest_sessions; scheduling
is deliberately out of scope pending review.
"""

from xnch.memory.session_ingest.ingestor import IngestReport, ingest_sessions
from xnch.memory.session_ingest.models import (
    FactEntity,
    FactTriple,
    SessionDigest,
    SessionSummary,
)
from xnch.memory.session_ingest.parser import parse_session
from xnch.memory.session_ingest.redactor import redact_text

__all__ = [
    "FactEntity",
    "FactTriple",
    "IngestReport",
    "SessionDigest",
    "SessionSummary",
    "ingest_sessions",
    "parse_session",
    "redact_text",
]
