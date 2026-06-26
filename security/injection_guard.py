from __future__ import annotations

import re
from dataclasses import dataclass, field

from xnch.audit.event_log import EventLog

INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r'ignore previous instructions', re.I),
    re.compile(r'forget your (system prompt|character|identity)', re.I),
    re.compile(r'you are now', re.I),
    re.compile(r'your new (instructions|role|persona)', re.I),
    re.compile(r'disregard.*above', re.I),
    re.compile(r'act as (?!(Nexi|nexi))', re.I),
    re.compile(r'jailbreak', re.I),
    re.compile(r'DAN mode', re.I),
    re.compile(r'pretend (you|that)', re.I),
]


@dataclass
class InjectionResult:
    is_clean: bool
    matched_patterns: list[str] = field(default_factory=list)
    risk_score: float = 0.0


def scan_input(text: str, event_log: EventLog | None = None) -> InjectionResult:
    matched: list[str] = []
    for pattern in INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            matched.append(m.group(0))

    if not matched:
        return InjectionResult(is_clean=True, risk_score=0.0)

    risk_score = len(matched) / len(INJECTION_PATTERNS)

    if risk_score > 0.1 and event_log is not None:
        event_log.emit(
            "injection_guard", "security",
            "injection_attempt",
            level="WARNING",
            data={
                "matched_patterns": matched,
                "risk_score": risk_score,
                "text_preview": text[:200],
            },
        )

    is_clean = risk_score == 0.0
    return InjectionResult(is_clean=is_clean, matched_patterns=matched, risk_score=risk_score)
