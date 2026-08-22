"""Tests for session-ingest secret redaction."""

from __future__ import annotations

from xnch.memory.session_ingest.redactor import redact_text


def _clean(text: str) -> str:
    cleaned, _ = redact_text(text)
    return cleaned


def test_openai_style_key_redacted():
    text = "use this key sk-proj-4f8aBcD2eF9gH1iJ3kL5mN7oP9qR1sT3uV5wX7yZ for calls"
    assert "sk-proj-" not in _clean(text)
    assert "[REDACTED:" in _clean(text)


def test_github_tokens_redacted():
    text = "ghp_1234567890abcdefghijklmnopqrstuvwxyz and github_pat_11AAAAAAA0abcdefghijklmnopqrstuv"
    cleaned = _clean(text)
    assert "ghp_" not in cleaned
    assert "github_pat_" not in cleaned


def test_aws_credentials_redacted():
    text = (
        "AKIAIOSFODNN7EXAMPLE with secret "
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    )
    cleaned = _clean(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in cleaned
    assert "wJalrXUtnFEMI" not in cleaned


def test_bearer_and_jwt_redacted():
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    text = f"Authorization: Bearer {jwt}"
    cleaned = _clean(text)
    assert "eyJ" not in cleaned
    assert "Bearer [REDACTED" in cleaned or "[REDACTED:" in cleaned


def test_private_key_block_redacted():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA7\n-----END RSA PRIVATE KEY-----"
    cleaned = _clean(text)
    assert "MIIEpAIBAAKCAQEA7" not in cleaned


def test_connection_string_password_redacted():
    text = "postgresql://xnch:s3cretPw@localhost:5432/xnch and redis://:redispass@localhost:6379/0"
    cleaned = _clean(text)
    assert "s3cretPw" not in cleaned
    assert "redispass" not in cleaned
    assert "postgresql://xnch:" in cleaned
    assert "localhost:5432" in cleaned


def test_generic_key_value_assignment_redacted():
    text = 'api_key="a1b2c3d4e5f6g7h8" and PASSWORD=hunter2secret'
    cleaned = _clean(text)
    assert "a1b2c3d4e5f6g7h8" not in cleaned
    assert "hunter2secret" not in cleaned


def test_slack_and_google_keys_redacted():
    slack = "-".join(["xoxb", "123456789", "0987654321", "AbCdEfGhIjKl"])
    google = "".join(["AIza", "Sy", "A", "1" * 10, "a" * 22])
    text = f"{slack} and {google}"
    cleaned, _ = redact_text(text)
    assert "xoxb-" not in cleaned
    assert "AIza" not in cleaned


def test_sha256_checksum_not_redacted():
    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    text = f"sha256 {sha} matches upstream release checksum"
    cleaned, findings = redact_text(text)
    assert sha in cleaned
    assert findings == {}


def test_clean_text_untouched():
    text = "Refactored select_decision in nexi/pipeline/evaluator.py; all tests pass."
    cleaned, findings = redact_text(text)
    assert cleaned == text
    assert findings == {}


def test_findings_count_occurrences_by_type():
    key = "sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    text = f"{key}\n{key}"
    _, findings = redact_text(text)
    assert sum(findings.values()) == 2
