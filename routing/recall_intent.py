"""Recall-intent parsing for Nexi chat (shared by gateway and voice)."""

from __future__ import annotations

import re

_RECALL_RE = re.compile(
    r"^\s*(?:/recall|recall memory|memory recall)\s+(.+?)\s*$", re.IGNORECASE
)


def recall_query(text: str) -> str | None:
    match = _RECALL_RE.match(text)
    return match.group(1) if match else None
