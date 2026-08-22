"""Tests for bi-temporal extensions to the Kuzu GraphStore."""

from __future__ import annotations

import kuzu
import pytest

from xnch.memory.graph_store import GraphStore


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=tmp_path / "graph")
    s.connect()
    yield s
    s.close()


def _columns(conn: kuzu.Connection, table: str) -> set[str]:
    result = conn.execute(f"CALL table_info('{table}') RETURN *")
    cols = set()
    while result.has_next():
        cols.add(result.get_next()[1])
    return cols


def test_fresh_schema_has_bitemporal_columns(store):
    conn = store._conn
    for table in ("entities", "relations"):
        cols = _columns(conn, table)
        assert {"valid_from", "invalidated_at", "source"} <= cols


def test_old_schema_db_migrated_without_data_loss(tmp_path):
    db_dir = tmp_path / "legacy"
    db_dir.mkdir()
    db = kuzu.Database(str(db_dir / "graph.kuzu"))
    conn = kuzu.Connection(db)
    conn.execute(
        """CREATE NODE TABLE entities (
               entity_id STRING PRIMARY KEY, name STRING, type STRING,
               created_at TIMESTAMP DEFAULT current_timestamp())"""
    )
    conn.execute(
        """CREATE REL TABLE relations (FROM entities TO entities,
               rel_type STRING, confidence DOUBLE,
               created_at TIMESTAMP DEFAULT current_timestamp())"""
    )
    conn.execute("CREATE (a:entities {entity_id:'x', name:'X', type:'tool'})")
    conn.close()
    db.close()

    store = GraphStore(db_path=db_dir / "graph.kuzu")
    store.connect()
    try:
        assert {"valid_from", "invalidated_at", "source"} <= _columns(
            store._conn, "entities"
        )
        ent = store.get_entity("x")
        assert ent is not None and ent["name"] == "X"
    finally:
        store.close()


async def test_upsert_relation_persists_valid_from_and_source(store):
    from datetime import datetime, timezone

    store.upsert_entity(id="a", name="Alpha", type_="tech")
    store.upsert_entity(id="b", name="Beta", type_="tech")
    when = datetime(2026, 8, 1, tzinfo=timezone.utc)
    await store.upsert_relation(
        from_id="a", to_id="b", rel_type="chosen_over",
        confidence=0.9, valid_from=when, source="opencode:ses_x",
    )
    rows = store.list_relations()
    assert len(rows) == 1
    assert rows[0]["source"] == "opencode:ses_x"
    expected = when.astimezone(timezone.utc).replace(tzinfo=None).isoformat()
    assert rows[0]["valid_from"] == expected
    assert rows[0]["invalidated_at"] is None


async def test_legacy_upsert_relation_call_still_works(store):
    from datetime import datetime, timedelta, timezone

    store.upsert_entity(id="c", name="Gamma", type_="tech")
    store.upsert_entity(id="d", name="Delta", type_="tech")
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    await store.upsert_relation(
        from_id="c", to_id="d", rel_type="related_to", confidence=0.5
    )
    row = store.list_relations()[0]
    assert row["source"] is None
    vf = datetime.fromisoformat(row["valid_from"])
    assert vf >= before - timedelta(seconds=5)


async def test_reingest_updates_confidence_keeps_source(store):
    from datetime import datetime, timezone

    store.upsert_entity(id="e", name="Eps", type_="t")
    store.upsert_entity(id="f", name="Foo", type_="t")
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await store.upsert_relation(from_id="e", to_id="f", rel_type="r",
                                confidence=0.4, valid_from=when, source="opencode:s1")
    await store.upsert_relation(from_id="e", to_id="f", rel_type="r",
                                confidence=0.7, valid_from=when, source="opencode:s2")
    rows = [r for r in store.list_relations() if r["rel_type"] == "r"]
    assert len(rows) == 1
    assert rows[0]["confidence"] == 0.7
