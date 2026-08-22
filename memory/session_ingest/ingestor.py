"""Orchestrator: parse OpenCode sessions -> summarize -> episodic + semantic.

Ordering guarantee: every string passes redact_text before any store call.
Idempotency: sessions already marked SUCCEEDED in the ledger are skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from xnch.memory.graph_store import GraphStore
from xnch.memory.pg_episodic_store import PgEpisodicStore
from xnch.memory.session_ingest.models import (
    SessionDigest,
    SessionSummary,
)
from xnch.memory.session_ingest.parser import iter_session_refs, parse_session
from xnch.memory.session_ingest.redactor import redact_text
from xnch.memory.session_ingest.summarizer import summarize_session

logger = logging.getLogger(__name__)

_EPISODE_IMPORTANCE = 1.5


class IngestReport(BaseModel):
    scanned: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    facts_written: int = 0
    errors: dict[str, str] = Field(default_factory=dict)


async def ingest_sessions(
    db_path: Path,
    pg: PgEpisodicStore,
    graph: GraphStore | None,
    *,
    session_ids: list[str] | None = None,
    updated_since_ms: int | None = None,
    directories: list[str] | None = None,
    summarize=None,
    dry_run: bool = False,
    limit: int | None = None,
) -> IngestReport:
    """Ingest OpenCode sessions into the episodic and semantic tiers."""
    summarize_fn = summarize or summarize_session
    refs = iter_session_refs(
        db_path, updated_since_ms=updated_since_ms, directories=directories
    )
    if session_ids:
        wanted = set(session_ids)
        refs = [r for r in refs if r.session_id in wanted]
    if limit is not None:
        refs = refs[: max(limit, 0)]

    completed = await pg.ledger_completed_ids()
    report = IngestReport(scanned=len(refs))

    for ref in refs:
        if ref.session_id in completed:
            report.skipped += 1
            continue
        try:
            digest = _require_digest(db_path, ref.session_id)
            digest = _redact_digest(digest)
            summary = await summarize_fn(digest)
            summary = _redact_summary(summary)
            if dry_run:
                report.succeeded += 1
                continue
            facts_written = await _write_facts(graph, digest, summary)
            episode_id = await _write_episode(pg, digest, summary)
            await pg.ledger_mark_done(
                digest.session_id, episode_id, facts_count=facts_written
            )
            report.facts_written += facts_written
            report.succeeded += 1
        except Exception as exc:
            logger.warning("Session %s failed ingestion: %s", ref.session_id, exc)
            if not dry_run:
                await pg.ledger_mark_failed(ref.session_id, str(exc))
            report.failed += 1
            report.errors[ref.session_id] = str(exc)

    logger.info(
        "Session ingest: %d scanned, %d ok, %d failed, %d skipped",
        report.scanned, report.succeeded, report.failed, report.skipped,
    )
    return report


def _require_digest(db_path: Path, session_id: str) -> SessionDigest:
    digest = parse_session(db_path, session_id)
    if digest is None:
        raise ValueError(f"session {session_id} not found")
    return digest


def _redact_digest(digest: SessionDigest) -> SessionDigest:
    title, _ = redact_text(digest.title)
    goal, _ = redact_text(digest.goal)
    transcript, _ = redact_text(digest.transcript_digest)
    return digest.model_copy(
        update={"title": title, "goal": goal, "transcript_digest": transcript}
    )


def _redact_summary(summary: SessionSummary) -> SessionSummary:
    text, _ = redact_text(summary.summary)
    decisions = [redact_text(d)[0] for d in summary.decisions]
    outcome = summary.outcome
    facts = []
    for f in summary.facts:
        subject = f.subject.model_copy(
            update={
                "id": redact_text(f.subject.id)[0],
                "name": redact_text(f.subject.name)[0],
            }
        )
        obj = f.object.model_copy(
            update={
                "id": redact_text(f.object.id)[0],
                "name": redact_text(f.object.name)[0],
            }
        )
        facts.append(f.model_copy(update={"subject": subject, "object": obj}))
    return summary.model_copy(
        update={
            "summary": text,
            "decisions": decisions,
            "outcome": outcome,
            "facts": facts,
        }
    )


async def _write_episode(
    pg: PgEpisodicStore, digest: SessionDigest, summary: SessionSummary
) -> str:
    rendered = _render_summary_text(summary)
    return await pg.store_session_episode(
        raw_text=digest.transcript_digest,
        summary=rendered,
        importance=_EPISODE_IMPORTANCE,
        timestamp=digest.ended_at,
    )


def _render_summary_text(summary: SessionSummary) -> str:
    parts = [summary.summary]
    if summary.decisions:
        parts.append("Decisions:\n" + "\n".join(f"- {d}" for d in summary.decisions))
    if summary.outcome:
        parts.append(f"Outcome: {summary.outcome}")
    return "\n\n".join(parts)


async def _write_facts(
    graph: GraphStore, digest: SessionDigest, summary: SessionSummary
) -> int:
    valid_from = digest.ended_at or digest.started_at
    source = f"opencode:{digest.session_id}"
    written = 0
    for fact in summary.facts:
        graph.upsert_entity(
            id=fact.subject.id, name=fact.subject.name, type_=fact.subject.type
        )
        graph.upsert_entity(
            id=fact.object.id, name=fact.object.name, type_=fact.object.type
        )
        await graph.upsert_relation(
            from_id=fact.subject.id,
            to_id=fact.object.id,
            rel_type=fact.relation,
            confidence=0.8,
            valid_from=valid_from,
            source=source,
        )
        written += 1
    return written
