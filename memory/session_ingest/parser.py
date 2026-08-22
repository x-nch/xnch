"""Read-only parser for the OpenCode SQLite session store.

Reads ``~/.local/share/opencode/opencode.db`` (session/message/part tables,
JSON part payloads) and produces SessionDigest models. The database is opened
with SQLite URI read-only mode so a live WAL database is never mutated.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from xnch.memory.session_ingest.models import SessionDigest

_FILE_TOOLS = ("edit", "write", "read")
_TRANSCRIPT_ITEM_LIMIT = 2000
_TRANSCRIPT_BUDGET = 12000


class SessionRef(BaseModel):
    session_id: str
    title: str
    directory: str
    time_updated: int


def default_db_path() -> Path:
    return Path.home() / ".local/share/opencode/opencode.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def iter_session_refs(
    db_path: Path,
    *,
    updated_since_ms: int | None = None,
    directories: list[str] | None = None,
) -> list[SessionRef]:
    query = (
        "SELECT id, title, directory, time_updated FROM session "
        "WHERE parent_id IS NULL"
    )
    params: list[object] = []
    if updated_since_ms is not None:
        query += " AND time_updated >= ?"
        params.append(updated_since_ms)
    query += " ORDER BY time_created ASC"
    conn = _connect(db_path)
    try:
        rows = [dict(r) for r in conn.execute(query, params)]
    finally:
        conn.close()
    refs = [
        SessionRef(
            session_id=r["id"],
            title=r["title"] or "",
            directory=r["directory"] or "",
            time_updated=int(r["time_updated"]),
        )
        for r in rows
    ]
    if directories:
        refs = [
            r for r in refs if any(r.directory.startswith(d) for d in directories)
        ]
    return refs


def parse_session(db_path: Path, session_id: str) -> SessionDigest | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM session WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        parts = _load_parts(conn, session_id)
    finally:
        conn.close()
    return _build_digest(row, parts)


def _load_parts(
    conn: sqlite3.Connection, session_id: str
) -> list[tuple[str, dict]]:
    rows = conn.execute(
        """SELECT p.data AS data, m.data AS msg_data
           FROM part p JOIN message m ON m.id = p.message_id
           WHERE p.session_id = ?
           ORDER BY p.time_created ASC, p.id ASC""",
        (session_id,),
    ).fetchall()
    loaded: list[tuple[str, dict]] = []
    for r in rows:
        try:
            payload = json.loads(r["data"])
            message = json.loads(r["msg_data"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        role = str(message.get("role", "")) if isinstance(message, dict) else ""
        loaded.append((role, payload))
    return loaded


def _build_digest(session_row: sqlite3.Row, parts: list[tuple[str, dict]]) -> SessionDigest:
    goal = ""
    files_touched: list[str] = []
    tools_used: dict[str, int] = {}
    transcript: list[str] = []

    for role, part in parts:
        ptype = str(part.get("type", ""))
        if ptype == "text":
            text = str(part.get("text") or "").strip()
            if not text:
                continue
            transcript.append(_clip(f"{role or 'unknown'}: {text}"))
            if role == "user" and not goal:
                goal = text
        elif ptype == "tool":
            tool = str(part.get("tool") or "unknown")
            tools_used[tool] = tools_used.get(tool, 0) + 1
            state = part.get("state")
            state_input = state.get("input") if isinstance(state, dict) else None
            if not isinstance(state_input, dict):
                continue
            file_path = state_input.get("filePath") or state_input.get("path")
            if tool in _FILE_TOOLS and isinstance(file_path, str) and file_path:
                if file_path not in files_touched:
                    files_touched.append(file_path)
            command = state_input.get("command")
            if tool == "bash" and isinstance(command, str) and command.strip():
                transcript.append(_clip(f"$ {command.strip()}"))

    return SessionDigest(
        session_id=str(session_row["id"]),
        title=str(session_row["title"] or ""),
        directory=str(session_row["directory"] or ""),
        project_id=str(session_row["project_id"] or ""),
        agent=str(session_row["agent"] or ""),
        model=str(session_row["model"] or ""),
        started_at=_ms_to_dt(int(session_row["time_created"])),
        ended_at=_ms_to_dt(int(session_row["time_updated"])),
        goal=goal or str(session_row["title"] or ""),
        files_touched=files_touched,
        tools_used=tools_used,
        transcript_digest=_join_transcript(transcript),
    )


def _clip(text: str, limit: int = _TRANSCRIPT_ITEM_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[clipped]"


def _join_transcript(lines: list[str]) -> str:
    out: list[str] = []
    total = 0
    for line in lines:
        if total + len(line) > _TRANSCRIPT_BUDGET:
            out.append("...[truncated]")
            break
        out.append(line)
        total += len(line) + 1
    return "\n".join(out)


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
