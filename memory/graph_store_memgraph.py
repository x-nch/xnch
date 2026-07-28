"""Memgraph-backed semantic graph store — replaces agentmemory-based graph_store.py.

Uses the neo4j driver directly (Memgraph is Bolt-protocol compatible).
"""
from __future__ import annotations

import os
from typing import Any

from neo4j import GraphDatabase


class MemgraphGraphStore:
    """Graph store backed by Memgraph for native Cypher traversal."""

    def __init__(
        self,
        uri: str | None = None,
        username: str = "",
        password: str = "",
    ) -> None:
        self._uri = uri or os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
        self._username = username or os.environ.get("MEMGRAPH_USERNAME", "")
        self._password = password or os.environ.get("MEMGRAPH_PASSWORD", "")
        self._driver = None

    def connect(self) -> None:
        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._username, self._password)
        )

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self.connect()
        return self._driver

    def _query(self, cypher: str, params: dict | None = None) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def _run(self, cypher: str, params: dict | None = None) -> None:
        with self.driver.session() as session:
            session.run(cypher, params or {})

    def upsert_entity(self, id: str, name: str, type_: str) -> None:
        self._run(
            """
            MERGE (e:Entity {id: $id})
            SET e.name = $name, e.type = $type, e.updated_at = datetime()
            """,
            {"id": id, "name": name, "type": type_},
        )

    async def upsert_relation(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        confidence: float,
    ) -> None:
        self._run(
            """
            MATCH (a:Entity {id: $from_id})
            MATCH (b:Entity {id: $to_id})
            MERGE (a)-[r:RELATES_TO {type: $rel_type}]->(b)
            SET r.confidence = $confidence, r.updated_at = datetime()
            """,
            {
                "from_id": from_id,
                "to_id": to_id,
                "rel_type": rel_type,
                "confidence": confidence,
            },
        )

    def query_entity_connections(self, entity_id: str) -> list[dict[str, Any]]:
        return self._query(
            """
            MATCH (e:Entity {id: $id})-[r:RELATES_TO]-(other:Entity)
            RETURN other.id AS connected_id, other.name AS connected_name,
                   other.type AS connected_type, r.type AS rel_type,
                   r.confidence AS confidence
            ORDER BY r.confidence DESC
            LIMIT 50
            """,
            {"id": entity_id},
        )

    def get_entity_by_name(self, name: str) -> dict[str, Any] | None:
        rows = self._query(
            """
            MATCH (e:Entity)
            WHERE e.name = $name
            RETURN e.id AS entity_id, e.name AS name, e.type AS type
            LIMIT 1
            """,
            {"name": name},
        )
        return {"metadata": rows[0]} if rows else None

    def get_entity_by_id(self, entity_id: str) -> dict[str, Any] | None:
        rows = self._query(
            """
            MATCH (e:Entity {id: $id})
            RETURN e.id AS entity_id, e.name AS name, e.type AS type
            """,
            {"id": entity_id},
        )
        return {"metadata": rows[0]} if rows else None

    def find_related_entities(
        self, entity_id: str, max_depth: int = 2
    ) -> list[dict[str, Any]]:
        """Graph-native BFS traversal for impact radius / deep connections.

        Uses Memgraph-native *BFS for efficient shortest-path expansion.
        """
        return self._query(
            """
            MATCH (e:Entity {id: $id})
            WITH e
            MATCH path = (e)-[:RELATES_TO *BFS..$depth]-(other:Entity)
            WHERE other.id <> $id
            RETURN DISTINCT other.id AS entity_id, other.name AS name,
                   other.type AS type, length(path) AS depth
            ORDER BY depth ASC
            LIMIT 30
            """,
            {"id": entity_id, "depth": max_depth},
        )

    def analyze_graph(self) -> None:
        """Run ANALYZE GRAPH to update Memgraph query planner statistics.

        Should be called after bulk data loads.
        """
        self._run("ANALYZE GRAPH")

    def persist_episode(
        self,
        episode_id: str,
        intent_class: str,
        action_type: str,
        entity_id: str | None = None,
        outcome: str = "",
    ) -> None:
        self._run(
            """
            MERGE (ep:Episode {id: $id})
            SET ep.intent_class = $intent_class, ep.action_type = $action_type,
                ep.outcome = $outcome, ep.created_at = datetime()
            """,
            {
                "id": episode_id,
                "intent_class": intent_class,
                "action_type": action_type,
                "outcome": outcome,
            },
        )
        if entity_id:
            self._run(
                """
                MATCH (e:Entity {id: $entity_id})
                MATCH (ep:Episode {id: $episode_id})
                MERGE (e)-[:MENTIONED_IN]->(ep)
                """,
                {"entity_id": entity_id, "episode_id": episode_id},
            )

    def persist_pattern(
        self,
        pattern_id: str,
        context_signature: str,
        success_rate: float,
        confidence: float,
        entity_id: str | None = None,
    ) -> None:
        self._run(
            """
            MERGE (p:Pattern {id: $id})
            SET p.context_signature = $sig, p.success_rate = $sr,
                p.confidence = $conf, p.updated_at = datetime()
            """,
            {
                "id": pattern_id,
                "sig": context_signature,
                "sr": success_rate,
                "conf": confidence,
            },
        )
        if entity_id:
            self._run(
                """
                MATCH (p:Pattern {id: $pattern_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (p)-[:APPLIES_TO]->(e)
                """,
                {"pattern_id": pattern_id, "entity_id": entity_id},
            )

    def get_patterns_for_entity(
        self, entity_type: str, context_signature: str | None = None
    ) -> list[dict[str, Any]]:
        query = """
            MATCH (p:Pattern)-[:APPLIES_TO]->(e:Entity)
            WHERE e.type = $entity_type
        """
        params: dict[str, Any] = {"entity_type": entity_type}
        if context_signature:
            query += " AND p.context_signature = $sig"
            params["sig"] = context_signature
        query += """
            RETURN p.id AS id, p.context_signature AS context_signature,
                   p.success_rate AS success_rate, p.confidence AS confidence
            ORDER BY p.confidence DESC
            LIMIT 10
        """
        return self._query(query, params)
