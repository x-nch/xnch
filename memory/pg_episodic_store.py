"""PostgreSQL + pgvector episodic store — Layer 2 memory.
Backed by agentmemory (ChromaDB) with pgvector-compatible schema semantics."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

from agentmemory import create_memory, get_memories, get_memory, search_memory, update_memory


CATEGORY = "episodes"


class PgEpisodicStore:
    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn
        self._pool = None

    async def connect(self) -> None:
        pass

    async def store_episode(
        self,
        type_: str,
        raw_text: str | None = None,
        summary: str | None = None,
        embedding: list[float] | None = None,
        importance: float = 1.0,
    ) -> str:
        now = datetime.now(timezone.utc)
        memory_id = str(uuid.uuid4())
        m = {
            "type": type_,
            "raw_text": raw_text or "",
            "summary": summary or "",
            "importance": str(importance),
            "recall_count": "0",
            "last_recalled": "",
            "archived": "False",
            "timestamp": now.isoformat(),
        }
        create_memory(
            CATEGORY,
            raw_text or summary or "",
            id=memory_id,
            metadata=m,
        )
        return memory_id

    async def retrieve_similar(
        self,
        embedding: list[float] | None = None,
        query_text: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        if embedding is None:
            if query_text:
                results = search_memory(CATEGORY, query_text, n_results=top_k, filter_metadata={"archived": "False"})
                processed = []
                for r in results:
                    sim = 1.0 - r.get("distance", 0.0)
                    if sim < min_score:
                        continue
                    metadata = r.get("metadata", {})
                    processed.append({
                        "id": r["id"],
                        "type": metadata.get("type", ""),
                        "summary": metadata.get("summary", ""),
                        "raw_text": metadata.get("raw_text", ""),
                        "importance": float(metadata.get("importance", 1.0)),
                        "recall_count": int(metadata.get("recall_count", 0)),
                        "last_recalled": metadata.get("last_recalled", None),
                        "timestamp": metadata.get("timestamp", ""),
                        "decay_score": float(metadata.get("decay_score", 1.0)),
                        "archived": metadata.get("archived", "False") == "True",
                        "similarity": sim,
                    })
                return processed
            else:
                memories = get_memories(CATEGORY, n_results=max(50, top_k * 10))
                return [
                    {
                        "id": r["id"],
                        "timestamp": r["metadata"].get("timestamp", ""),
                        "type": r["metadata"].get("type", ""),
                        "raw_text": r["metadata"].get("raw_text", ""),
                        "summary": r["metadata"].get("summary", ""),
                        "embedding": r.get("embedding"),
                        "importance": float(r["metadata"].get("importance", 1.0)),
                        "recall_count": int(r["metadata"].get("recall_count", 0)),
                        "last_recalled": r["metadata"].get("last_recalled", None),
                        "decay_score": float(r["metadata"].get("decay_score", 1.0)),
                        "archived": r["metadata"].get("archived", "False") == "True",
                        "similarity": 1.0,
                    }
                    for r in sorted(memories, key=lambda m: m.get("metadata", {}).get("created_at", ""), reverse=True)[:top_k]
                ]
        search_query = query_text or ''
        results = search_memory(CATEGORY, search_query, n_results=top_k, filter_metadata={"archived": "False"})
        processed = []
        for r in results:
            dist = r.get("distance", 1.0)
            sim = 1.0 - dist
            if sim < min_score:
                continue
            processed.append({
                "id": r["id"],
                "timestamp": r["metadata"].get("timestamp", ""),
                "type": r["metadata"].get("type", ""),
                "raw_text": r["metadata"].get("raw_text", ""),
                "summary": r["metadata"].get("summary", ""),
                "embedding": r.get("embedding"),
                "importance": float(r["metadata"].get("importance", 1.0)),
                "recall_count": int(r["metadata"].get("recall_count", 0)),
                "last_recalled": r["metadata"].get("last_recalled", None),
                "decay_score": float(r["metadata"].get("decay_score", 1.0)),
                "archived": r["metadata"].get("archived", "False") == "True",
                "similarity": sim,
            })
        return processed

    async def bump_recall(self, id: str) -> None:
        mem = self._get_memory_by_id(id)
        if not mem:
            return
        meta = dict(mem["metadata"])
        rc = int(meta.get("recall_count", 0)) + 1
        meta["recall_count"] = str(rc)
        meta["last_recalled"] = datetime.now(timezone.utc).isoformat()
        update_memory(CATEGORY, id, metadata=meta)

    async def list_recent(self, hours: int = 24) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        all_mem = get_memories(CATEGORY, n_results=5000)
        results = []
        for r in all_mem:
            ts_str = r["metadata"].get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except Exception:
                continue
            if ts < cutoff:
                continue
            results.append({
                "id": r["id"],
                "timestamp": ts_str,
                "type": r["metadata"].get("type", ""),
                "raw_text": r["metadata"].get("raw_text", ""),
                "summary": r["metadata"].get("summary", ""),
                "importance": float(r["metadata"].get("importance", 1.0)),
                "recall_count": int(r["metadata"].get("recall_count", 0)),
                "last_recalled": r["metadata"].get("last_recalled", None),
                "decay_score": float(r["metadata"].get("decay_score", 1.0)),
                "archived": r["metadata"].get("archived", "False") == "True",
            })
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results

    async def execute(self, query: str, *args: Any) -> str:
        raise NotImplementedError("execute() stub removed — use store_episode or direct agentmemory calls")

    async def fetchval(self, query: str, *args: Any) -> Any:
        raise NotImplementedError("fetchval() stub removed — use agentmemory API directly")

    def _get_memory_by_id(self, id: str) -> dict | None:
        return get_memory(CATEGORY, id)

    async def store_decision_episode(
        self,
        decision_id: str,
        intent_class: str,
        action_type: str,
        entity_class: str,
        actor_role: str,
        context_snapshot: dict[str, Any] | None = None,
        scores_json: str | None = None,
        generation_path: str = "MODEL",
    ) -> str:
        """Stub — v0 uses SQLite EpisodicStore for decision episodes."""
        return str(uuid.uuid4())

    async def complete_decision_episode(
        self,
        decision_id: str,
        outcome: str,
        prediction_delta: float | None = None,
        early_reextraction_flag: bool = False,
    ) -> str | None:
        """Stub — v0 uses SQLite EpisodicStore for decision episodes."""
        return None

    async def fetch_decision_episodes_since(
        self,
        since: datetime,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        all_mem = get_memories(CATEGORY, n_results=limit)
        episodes = []
        for m in all_mem:
            ts_str = m["metadata"].get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except Exception:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < since:
                continue
            episodes.append(m)
        episodes.sort(key=lambda x: x["metadata"].get("timestamp", ""))
        return episodes[:limit]

    async def fetch_decision_episodes_with_scores(
        self,
        since: datetime,
    ) -> list[dict[str, Any]]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT episode_id, decision_id, intent_class, action_type,
                          entity_class, actor_role, outcome, prediction_delta,
                          scores_json, created_at, completed_at
                   FROM decision_episodes
                   WHERE created_at >= $1 AND outcome IS NOT NULL AND scores_json IS NOT NULL
                    ORDER BY created_at DESC""",
                since,
            )
        return [dict(r) for r in rows]

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
