"""Decision Ledger — SHA-256 chained, append-only, tamper-evident."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class DecisionLedger:
    def __init__(self, ledger_path: Path) -> None:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = ledger_path
        self._prev_hash: str = self._read_last_hash()

    def _read_last_hash(self) -> str:
        """Read the hash of the last entry; genesis hash if ledger is empty."""
        if not self._path.exists():
            return "sha256:" + "0" * 64
        last_hash = "sha256:" + "0" * 64
        try:
            with self._path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        last_hash = entry.get("hash", last_hash)
        except Exception:
            pass
        return last_hash

    def write(
        self,
        decision_id: str,
        trace_id: str,
        intent_hash: str,
        candidates_count: int,
        selected_option_id: str | None,
        scores: dict[str, Any],
        audit_ref: str,
    ) -> str:
        entry: dict[str, Any] = {
            "decision_id": decision_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "intent_hash": intent_hash,
            "candidates_count": candidates_count,
            "selected_option_id": selected_option_id,
            "scores": scores,
            "audit_ref": audit_ref,
            "prev_hash": self._prev_hash,
        }
        content = json.dumps(
            {k: v for k, v in entry.items()},
            sort_keys=True,
        )
        entry_hash = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
        entry["hash"] = entry_hash

        with self._path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

        self._prev_hash = entry_hash
        return audit_ref

    @staticmethod
    def verify_chain(ledger_path: Path) -> bool:
        if not ledger_path.exists():
            return True
        prev_entry = None
        with ledger_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if prev_entry:
                    if entry.get("prev_hash") != prev_entry.get("hash"):
                        return False
                    content = json.dumps(
                        {k: v for k, v in entry.items() if k != "hash"},
                        sort_keys=True,
                    )
                    expected = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
                    if entry.get("hash") != expected:
                        return False
                prev_entry = entry
        return True
