"""Semantic graph store — Layer 4 memory backed by Kuzu (embedded property graph).

Replaces the agentmemory/ChromaDB-backed graph store. Entities and typed
relations live in a Kuzu database (single-file, in-process, synchronous),
so the store keeps the same interface used by the pipeline:
get_entity_by_name / query_entity_connections / upsert_entity / upsert_relation.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import kuzu

from xnch.config import settings

ENTITIES_CATEGORY = "entities"
RELATIONS_CATEGORY = "relations"

_GRAPH_DIR = "graph.kuzu"

_SCHEMA = """
CREATE NODE TABLE IF NOT EXISTS entities (
    entity_id STRING PRIMARY KEY,
    name STRING,
    type STRING,
    created_at TIMESTAMP DEFAULT current_timestamp()
);

CREATE REL TABLE IF NOT EXISTS relations (
    FROM entities TO entities,
    rel_type STRING,
    confidence DOUBLE,
    created_at TIMESTAMP DEFAULT current_timestamp()
);
"""


class GraphStore:
    def __init__(
        self,
        db_path: Path | None = None,
        relationship_store: Any | None = None,
    ) -> None:
        self._relationship_store = relationship_store
        if db_path is None:
            self._dir = settings.base_dir / _GRAPH_DIR
        else:
            p = Path(db_path)
            self._dir = (p.parent if p.suffix else p) / _GRAPH_DIR
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        self._dir.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(self._dir))
        self._conn = kuzu.Connection(self._db)
        self._conn.execute(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn = None
            self._db = None

    # ------------------------------------------------------------------ #
    # Entities
    # ------------------------------------------------------------------ #

    def upsert_entity(self, id: str, name: str, type_: str) -> None:
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                """MERGE (e:entities {entity_id: $id})
                   ON CREATE SET e.name = $name, e.type = $type
                   ON MATCH SET e.name = $name, e.type = $type""",
                {"id": id, "name": name, "type": type_},
            )

    def get_entity_by_name(self, name: str) -> dict[str, Any] | None:
        if self._conn is None:
            return None
        with self._lock:
            result = self._conn.execute(
                """MATCH (e:entities)
                   WHERE lower(e.name) = lower($name)
                   RETURN e.entity_id, e.name, e.type
                   LIMIT 1""",
                {"name": name},
            )
            if not result.has_next():
                return None
            entity_id, entity_name, entity_type = result.get_next()
        return {
            "id": entity_id,
            "document": entity_name,
            "metadata": {
                "entity_id": entity_id,
                "name": entity_name,
                "type": entity_type,
            },
        }

    def _get_entity_direct(self, entity_id: str) -> dict[str, Any] | None:
        if self._conn is None:
            return None
        with self._lock:
            result = self._conn.execute(
                """MATCH (e:entities {entity_id: $id})
                   RETURN e.entity_id, e.name, e.type
                   LIMIT 1""",
                {"id": entity_id},
            )
            if not result.has_next():
                return None
            eid, ename, etype = result.get_next()
        return {
            "id": eid,
            "document": ename,
            "metadata": {"entity_id": eid, "name": ename, "type": etype},
        }

    def fetch_entities(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recently created entities (name for system prompts).

        Ordering is by created_at DESC (recency of creation), not insertion
        order, because Kuzu reserves the internal column `_id` for system use:
        referencing it raises "Binder exception: _id is reserved for system
        usage. External access is not allowed." (0.11.3), and even the id(e)
        accessor cannot be ordered by ("Order by INTERNAL_ID is not
        supported"). created_at is microsecond-precision, so sequential writes
        essentially never tie; entity_id (the PK) is added as a deterministic
        tiebreaker so the LIMIT window is stable across calls.
        """
        if self._conn is None:
            return []
        with self._lock:
            result = self._conn.execute(
                """MATCH (e:entities)
                   RETURN e.entity_id, e.name, e.type
                   ORDER BY e.created_at DESC, e.entity_id
                   LIMIT $limit""",
                {"limit": limit},
            )
            entities = []
            while result.has_next():
                eid, ename, etype = result.get_next()
                entities.append(
                    {
                        "id": eid,
                        "document": ename,
                        "metadata": {"entity_id": eid, "name": ename, "type": etype},
                    }
                )
        return entities

    # ------------------------------------------------------------------ #
    # Relations
    # ------------------------------------------------------------------ #

    async def upsert_relation(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        confidence: float,
    ) -> None:
        if self._conn is None:
            return
        with self._lock:
            exists = self._conn.execute(
                """MATCH (a:entities {entity_id: $f})-[r:relations {rel_type: $rt}]->(b:entities {entity_id: $t})
                   RETURN r.confidence""",
                {"f": from_id, "rt": rel_type, "t": to_id},
            )
            if exists.has_next():
                self._conn.execute(
                    """MATCH (a:entities {entity_id: $f})-[r:relations {rel_type: $rt}]->(b:entities {entity_id: $t})
                       SET r.confidence = $c""",
                    {"f": from_id, "rt": rel_type, "t": to_id, "c": float(confidence)},
                )
            else:
                self._conn.execute(
                    """MATCH (a:entities {entity_id: $f}), (b:entities {entity_id: $t})
                       CREATE (a)-[:relations {rel_type: $rt, confidence: $c}]->(b)""",
                    {"f": from_id, "rt": rel_type, "t": to_id, "c": float(confidence)},
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
        if self._conn is None:
            return []
        with self._lock:
            result = self._conn.execute(
                """MATCH (a:entities {entity_id: $id})-[r:relations]-(o:entities)
                   RETURN o.entity_id, o.name, o.type, r.rel_type, r.confidence""",
                {"id": entity_id},
            )
            rows = []
            while result.has_next():
                connected_id, connected_name, connected_type, rel_type, confidence = result.get_next()
                rows.append({
                    "connected_id": connected_id,
                    "connected_name": connected_name,
                    "connected_type": connected_type,
                    "rel_type": rel_type,
                    "confidence": float(confidence),
                })
        return rows
