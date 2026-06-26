from __future__ import annotations

from xnch.security.injection_guard import scan_input, INJECTION_PATTERNS
from xnch.security.trust_model import TrustLevel, get_trust_level


QUARANTINE_TABLE = "quarantine_memories"


def validate_memory_write(
    content: str,
    actor_role: str,
    trust_level: TrustLevel,
) -> tuple[bool, str | None]:
    result = scan_input(content)
    if not result.is_clean:
        return False, "Content failed injection scan"

    if trust_level.value < TrustLevel.TRUSTED_AGENT.value:
        return False, f"Trust level {trust_level.name} ({actor_role}) cannot write to episodic store directly"

    return True, None
