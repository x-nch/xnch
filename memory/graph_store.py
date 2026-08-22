"""Semantic graph store — Layer 4 memory backed by Kuzu (embedded property graph).

Replaces the agentmemory/ChromaDB-backed graph store. Entities and typed
relations live in a Kuzu database (single-file, in-process, synchronous),
so the store keeps the same interface used by the pipeline:
get_entity_by_name / query_entity_connections / upsert_entity / upsert_relation.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import kuzu

from xnch.config import settings

if TYPE_CHECKING:
    from xnch.memory.graph_broadcaster import GraphBroadcaster

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
        broadcaster: GraphBroadcaster | None = None,
    ) -> None:
        self._relationship_store = relationship_store
        self._broadcaster = broadcaster
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
        self._publish_entity(id, name, type_)

    def _publish_entity(self, entity_id: str, name: str, type_: str) -> None:
        if self._broadcaster is None:
            return
        ent = self.get_entity(entity_id)
        self._broadcaster.publish(
            {
                "type": "entity",
                "entity": ent
                or {"entity_id": entity_id, "name": name, "type": type_, "created_at": None},
            }
        )
        stats = self.get_stats()
        self._broadcaster.publish({"type": "stats", **stats})

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
        self._publish_relation(from_id, to_id, rel_type, confidence)

    def _publish_relation(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        confidence: float,
    ) -> None:
        if self._broadcaster is None:
            return
        from_ent = self.get_entity(from_id)
        to_ent = self.get_entity(to_id)
        self._broadcaster.publish(
            {
                "type": "relation",
                "relation": {
                    "from_id": from_id,
                    "from_name": from_ent["name"] if from_ent else None,
                    "to_id": to_id,
                    "to_name": to_ent["name"] if to_ent else None,
                    "rel_type": rel_type,
                    "confidence": float(confidence),
                    "created_at": None,
                },
            }
        )
        stats = self.get_stats()
        self._broadcaster.publish({"type": "stats", **stats})

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

    # ------------------------------------------------------------------ #
    # Graph explorer API
    # ------------------------------------------------------------------ #

    def count_entities(
        self,
        type_filter: str | None = None,
        search: str | None = None,
    ) -> int:
        if self._conn is None:
            return 0
        clauses = ["MATCH (e:entities)"]
        params: dict[str, Any] = {}
        if type_filter:
            clauses.append("WHERE e.type = $type")
            params["type"] = type_filter
        if search:
            connector = "WHERE" if "WHERE" not in " ".join(clauses) else "AND"
            clauses.append(f"{connector} lower(e.name) CONTAINS lower($search)")
            params["search"] = search
        clauses.append("RETURN count(e)")
        with self._lock:
            result = self._conn.execute("\n".join(clauses), params)
            if not result.has_next():
                return 0
            return int(result.get_next()[0])

    def list_entities(
        self,
        *,
        type_filter: str | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        clauses = ["MATCH (e:entities)"]
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if type_filter:
            clauses.append("WHERE e.type = $type")
            params["type"] = type_filter
        if search:
            connector = "WHERE" if "WHERE" not in " ".join(clauses) else "AND"
            clauses.append(f"{connector} lower(e.name) CONTAINS lower($search)")
            params["search"] = search
        clauses.extend(
            [
                "RETURN e.entity_id, e.name, e.type, e.created_at",
                "ORDER BY e.created_at DESC, e.entity_id",
                "SKIP $offset LIMIT $limit",
            ]
        )
        with self._lock:
            result = self._conn.execute("\n".join(clauses), params)
            rows: list[dict[str, Any]] = []
            while result.has_next():
                eid, name, etype, created_at = result.get_next()
                rows.append(
                    {
                        "entity_id": eid,
                        "name": name,
                        "type": etype,
                        "created_at": _ts_to_iso(created_at),
                    }
                )
        return rows

    def list_relations(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        with self._lock:
            result = self._conn.execute(
                """MATCH (a:entities)-[r:relations]->(b:entities)
                   RETURN a.entity_id, a.name, b.entity_id, b.name,
                          r.rel_type, r.confidence, r.created_at
                   ORDER BY r.created_at DESC, a.entity_id, b.entity_id
                   SKIP $offset LIMIT $limit""",
                {"limit": limit, "offset": offset},
            )
            rows: list[dict[str, Any]] = []
            while result.has_next():
                from_id, from_name, to_id, to_name, rel_type, confidence, created_at = (
                    result.get_next()
                )
                rows.append(
                    {
                        "from_id": from_id,
                        "from_name": from_name,
                        "to_id": to_id,
                        "to_name": to_name,
                        "rel_type": rel_type,
                        "confidence": float(confidence),
                        "created_at": _ts_to_iso(created_at),
                    }
                )
        return rows

    def count_relations(self) -> int:
        if self._conn is None:
            return 0
        with self._lock:
            result = self._conn.execute(
                "MATCH ()-[r:relations]->() RETURN count(r)"
            )
            if not result.has_next():
                return 0
            return int(result.get_next()[0])

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        if self._conn is None:
            return None
        with self._lock:
            result = self._conn.execute(
                """MATCH (e:entities {entity_id: $id})
                   RETURN e.entity_id, e.name, e.type, e.created_at
                   LIMIT 1""",
                {"id": entity_id},
            )
            if not result.has_next():
                return None
            eid, name, etype, created_at = result.get_next()
        return {
            "entity_id": eid,
            "name": name,
            "type": etype,
            "created_at": _ts_to_iso(created_at),
        }

    def get_subgraph(
        self,
        entity_id: str,
        depth: int = 1,
        max_entities: int = 200,
    ) -> dict[str, Any]:
        """BFS neighborhood subgraph around entity_id."""
        if self._conn is None:
            return {"center_id": entity_id, "depth": depth, "entities": [], "relations": []}

        depth = max(1, min(depth, 2))
        center = self.get_entity(entity_id)
        if center is None:
            return {"center_id": entity_id, "depth": depth, "entities": [], "relations": []}

        visited: dict[str, dict[str, Any]] = {entity_id: center}
        frontier = {entity_id}
        all_relations: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()

        for _ in range(depth):
            if not frontier or len(visited) >= max_entities:
                break
            next_frontier: set[str] = set()
            for eid in frontier:
                if len(visited) >= max_entities:
                    break
                for edge in self._edges_for_entity(eid):
                    key = (edge["from_id"], edge["to_id"], edge["rel_type"])
                    if key not in seen_edges:
                        seen_edges.add(key)
                        all_relations.append(edge)
                    for nid in (edge["from_id"], edge["to_id"]):
                        if nid not in visited and len(visited) < max_entities:
                            ent = self.get_entity(nid)
                            if ent:
                                visited[nid] = ent
                                next_frontier.add(nid)
            frontier = next_frontier

        return {
            "center_id": entity_id,
            "depth": depth,
            "entities": list(visited.values()),
            "relations": all_relations,
        }

    def get_stats(self) -> dict[str, Any]:
        if self._conn is None:
            return {"entity_count": 0, "relation_count": 0, "types": {}}
        entity_count = self.count_entities()
        relation_count = self.count_relations()
        types: dict[str, int] = {}
        with self._lock:
            result = self._conn.execute(
                """MATCH (e:entities)
                   RETURN e.type, count(e)
                   ORDER BY count(e) DESC"""
            )
            while result.has_next():
                etype, count = result.get_next()
                types[str(etype)] = int(count)
        return {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "types": types,
        }

    def _edges_for_entity(self, entity_id: str) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        with self._lock:
            result = self._conn.execute(
                """MATCH (a:entities)-[r:relations]->(b:entities)
                   WHERE a.entity_id = $id OR b.entity_id = $id
                   RETURN a.entity_id, a.name, b.entity_id, b.name,
                          r.rel_type, r.confidence, r.created_at""",
                {"id": entity_id},
            )
            rows: list[dict[str, Any]] = []
            while result.has_next():
                from_id, from_name, to_id, to_name, rel_type, confidence, created_at = (
                    result.get_next()
                )
                rows.append(
                    {
                        "from_id": from_id,
                        "from_name": from_name,
                        "to_id": to_id,
                        "to_name": to_name,
                        "rel_type": rel_type,
                        "confidence": float(confidence),
                        "created_at": _ts_to_iso(created_at),
                    }
                )
        return rows


def _ts_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
