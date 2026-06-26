"""Semantic graph store — Layer 3 memory backed by agentmemory categories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentmemory import create_memory, search_memory, get_memories, update_memory


ENTITIES_CATEGORY = "entities"
RELATIONS_CATEGORY = "relations"


class GraphStore:
    def __init__(
        self,
        db_path: Path | None = None,
        relationship_store: Any | None = None,
    ) -> None:
        self._path = db_path
        self._relationship_store = relationship_store

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def upsert_entity(self, id: str, name: str, type_: str) -> None:
        existing = search_memory(ENTITIES_CATEGORY, name, n_results=5)
        match = next(
            (e for e in existing if e["metadata"].get("entity_id") == id),
            None,
        )
        if match:
            update_memory(
                ENTITIES_CATEGORY, match["id"],
                metadata={"entity_id": id, "name": name, "type": type_},
            )
        else:
            create_memory(
                ENTITIES_CATEGORY, name,
                id=id,
                metadata={"entity_id": id, "name": name, "type": type_},
            )

    async def upsert_relation(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        confidence: float,
    ) -> None:
        rel_text = f"{from_id} {rel_type} {to_id}"
        existing = search_memory(RELATIONS_CATEGORY, rel_text, n_results=5)
        match = next(
            (r for r in existing
             if r["metadata"].get("from_id") == from_id
             and r["metadata"].get("to_id") == to_id
             and r["metadata"].get("rel_type") == rel_type),
            None,
        )
        if match:
            update_memory(
                RELATIONS_CATEGORY, match["id"],
                metadata={
                    "from_id": from_id, "to_id": to_id,
                    "rel_type": rel_type, "confidence": str(confidence),
                },
            )
        else:
            create_memory(
                RELATIONS_CATEGORY, rel_text,
                metadata={
                    "from_id": from_id, "to_id": to_id,
                    "rel_type": rel_type, "confidence": str(confidence),
                },
            )
        if self._relationship_store is not None:
            await self._relationship_store.upsert_relationship(
                entity_a=from_id,
                entity_b=to_id,
                rel_type=rel_type,
                evidence=f"confidence={confidence}",
                strength=confidence,
            )

    def query_entity_connections(self, entity_id: str) -> list[dict[str, Any]]:
        all_rel = get_memories(RELATIONS_CATEGORY, n_results=5000)
        rows = []
        for r in all_rel:
            meta = r["metadata"]
            if meta.get("from_id") == entity_id or meta.get("to_id") == entity_id:
                other_id = meta["to_id"] if meta["from_id"] == entity_id else meta["from_id"]
                entity = self._get_entity_direct(other_id)
                rows.append({
                    "connected_id": other_id,
                    "connected_name": entity["metadata"].get("name", other_id) if entity else other_id,
                    "connected_type": entity["metadata"].get("type", "") if entity else "",
                    "rel_type": meta.get("rel_type", ""),
                    "confidence": float(meta.get("confidence", 0.0)),
                })
        return rows

    def get_entity_by_name(self, name: str) -> dict[str, Any] | None:
        results = search_memory(ENTITIES_CATEGORY, name, n_results=3)
        for r in results:
            if r["document"].lower() == name.lower():
                return r
        return results[0] if results else None

    def _get_entity_direct(self, entity_id: str) -> dict | None:
        results = get_memories(ENTITIES_CATEGORY, filter_metadata={"entity_id": entity_id}, n_results=1)
        return results[0] if results else None
