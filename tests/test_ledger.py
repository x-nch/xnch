"""Decision Ledger — SHA-256 chain integrity tests."""
from uuid import uuid4

import pytest

from xnch.audit.ledger import DecisionLedger


@pytest.fixture
def ledger(tmp_path):
    return DecisionLedger(tmp_path / "decisions.jsonl")


def _write(ledger: DecisionLedger, n: int = 3) -> list[str]:
    refs = []
    for i in range(n):
        ref = ledger.write(
            decision_id=str(uuid4()),
            trace_id=str(uuid4()),
            intent_hash=f"sha256:{'a' * 64}",
            candidates_count=5,
            selected_option_id=str(uuid4()),
            scores={"composite": 0.75},
            audit_ref=str(uuid4()),
        )
        refs.append(ref)
    return refs


def test_write_returns_audit_ref(ledger):
    ref = ledger.write(str(uuid4()), str(uuid4()), "sha256:" + "a" * 64, 5,
                       str(uuid4()), {}, str(uuid4()))
    assert ref  # non-empty


def test_chain_valid_after_writes(tmp_path):
    path = tmp_path / "decisions.jsonl"
    ledger = DecisionLedger(path)
    _write(ledger, 5)
    assert DecisionLedger.verify_chain(path) is True


def test_empty_ledger_is_valid(tmp_path):
    path = tmp_path / "empty.jsonl"
    assert DecisionLedger.verify_chain(path) is True


def test_prev_hash_links_entries(tmp_path):
    import json
    path = tmp_path / "decisions.jsonl"
    ledger = DecisionLedger(path)
    _write(ledger, 3)

    with path.open() as f:
        entries = [json.loads(line) for line in f if line.strip()]

    assert entries[1]["prev_hash"] == entries[0]["hash"]
    assert entries[2]["prev_hash"] == entries[1]["hash"]


def test_tampered_entry_fails_verification(tmp_path):
    import json
    path = tmp_path / "decisions.jsonl"
    ledger = DecisionLedger(path)
    _write(ledger, 3)

    lines = path.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["candidates_count"] = 999
    lines[1] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n")

    assert DecisionLedger.verify_chain(path) is False
