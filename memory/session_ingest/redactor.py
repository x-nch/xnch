"""Secret redaction for session-ingest payloads.

Every string that reaches Postgres or Kuzu passes through redact_text first.
Redaction replaces credential material with typed ``[REDACTED:<type>]``
placeholders and reports per-type hit counts so ingestion can log exposure.
"""

from __future__ import annotations

import re

_PLACEHOLDER = "[REDACTED:{typ}]"

_PATTERNS: list[tuple[str, re.Pattern[str], str | None]] = [
    (
        "bearer_token",
        re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]{16,}"),
        None,
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{10,}\b"),
        None,
    ),
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        None,
    ),
    (
        "github_token",
        re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})\b"
        ),
        None,
    ),
    (
        "aws_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        None,
    ),
    (
        "aws_key",
        re.compile(
            r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{20,}"
        ),
        None,
    ),
    (
        "slack_token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{15,}\b"),
        None,
    ),
    (
        "google_api_key",
        re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        None,
    ),
    (
        "openai_key",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
        None,
    ),
    (
        "connection_string",
        re.compile(
            r"((?:postgres(?:ql)?|redis|mysql|mongodb(?:\+srv)?)://[^:/@\s]*:)"
            r"([^@\s/]+)"
            r"(@)"
        ),
        "{g1}[REDACTED:connection_string]{g3}",
    ),
]

_KEY_CONTEXT = re.compile(
    r"(?i)\b(api[_-]?key|apikey|secret|token|passwd|password|pwd|credential)\b"
)
_HEX_RUN = re.compile(r"\b[a-f0-9]{40,}\b")
_HASH_CONTEXT = re.compile(r"(?i)\b(sha\d{2,3}|md5|checksum|digest|blake|hash)\b")

_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|apikey|secret|token|passwd|password|pwd|auth)\b(\s*[=:]\s*)['\"]?"
    r"(?!\[REDACTED)[^\s'\"]{8,}"
)


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Return (redacted_text, findings) where findings maps type -> hit count."""
    findings: dict[str, int] = {}

    def _sub(match: re.Match[str], typ: str, template: str | None) -> str:
        findings[typ] = findings.get(typ, 0) + 1
        if template is None:
            return _PLACEHOLDER.format(typ=typ)
        return template.format(g1=match.group(1), g3=match.group(3))

    result = text
    for typ, pattern, template in _PATTERNS:
        result = pattern.sub(lambda m, t=typ, tpl=template: _sub(m, t, tpl), result)

    def _sub_assignment(match: re.Match[str]) -> str:
        findings["key_value"] = findings.get("key_value", 0) + 1
        return f"{match.group(1)}{match.group(2)}{_PLACEHOLDER.format(typ='key_value')}"

    result = _ASSIGNMENT.sub(_sub_assignment, result)

    def _hex_guard(line: str) -> str:
        if not _HEX_RUN.search(line):
            return line
        if _HASH_CONTEXT.search(line):
            return line
        if not _KEY_CONTEXT.search(line):
            return line
        return _HEX_RUN.sub(
            lambda m: (_PLACEHOLDER.format(typ="hex_secret"), findings.__setitem__(
                "hex_secret", findings.get("hex_secret", 0) + 1))[0],
            line,
        )

    result = "\n".join(_hex_guard(line) for line in result.split("\n"))
    return result, findings
