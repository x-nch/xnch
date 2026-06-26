"""Contract 2: RS256 execution token issuance and jti replay protection."""
import time
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import jwt

from ..config import settings
from xnch.security.trust_model import TrustLevel, get_trust_level


_TOKEN_TTL_BY_TRUST: dict[TrustLevel, int] = {
    TrustLevel.SYSTEM: 7 * 86400,
    TrustLevel.OWNER: 86400,
    TrustLevel.TRUSTED_AGENT: 3600,
    TrustLevel.EXTERNAL_AGENT: 1800,
    TrustLevel.UNTRUSTED: 0,
}


@dataclass
class ExecutionTokenClaims:
    session_id: UUID
    decision_id: UUID
    trace_id: UUID
    actor_id: str
    actor_role: str
    action_type: str
    entity_class: str
    policy_version: str
    system_state_version: str


class _JtiSeenSet:
    """In-process replay protection. TTL-evicts entries after their exp passes."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}  # jti → exp unix-seconds

    def consume(self, jti: str, exp: float) -> bool:
        """Return True if jti is new (not replayed). Always adds to seen-set on first call."""
        self._evict()
        if jti in self._seen:
            return False
        self._seen[jti] = exp
        return True

    def _evict(self) -> None:
        now = time.time()
        stale = [k for k, exp in self._seen.items() if exp < now]
        for k in stale:
            del self._seen[k]


class TokenSigner:
    def __init__(self, private_pem: bytes) -> None:
        self._private_pem = private_pem

    def issue(self, claims: ExecutionTokenClaims) -> tuple[str, int]:
        """Issue a signed RS256 execution token. Returns (token, ttl_ms)."""
        now = int(time.time())
        trust_level = get_trust_level(claims.actor_role)
        ttl_s = _TOKEN_TTL_BY_TRUST.get(trust_level, 3600)
        ttl_ms = ttl_s * 1000
        payload = {
            "iss": "xnch",
            "sub": "execution_token",
            "jti": str(uuid4()),
            "iat": now,
            "exp": now + ttl_s,
            "role": claims.actor_role,
            "session_id": str(claims.session_id),
            "decision_id": str(claims.decision_id),
            "trace_id": str(claims.trace_id),
            "actor_id": claims.actor_id,
            "actor_role": claims.actor_role,
            "action_type": claims.action_type,
            "entity_class": claims.entity_class,
            "policy_version": claims.policy_version,
            "system_state_version": claims.system_state_version,
            "token_ttl_ms": ttl_ms,
            "trust_level": trust_level.name,
        }
        token = jwt.encode(payload, self._private_pem, algorithm="RS256")
        return token, ttl_ms


class TokenVerifier:
    """Verifies incoming auth tokens (HS256, shared secret) for xnch API callers."""

    def __init__(self, secret: str) -> None:
        self._secret = secret
        self._jti_seen = _JtiSeenSet()

    def verify_bearer(self, authorization: str) -> str | None:
        """Extract and verify actor_id from Authorization header.

        v0 accepts two formats:
          - 'actor:<actor_id>'  — plain actor reference (dev only)
          - 'Bearer <hs256-jwt>' — HS256 token with 'sub' = actor_id
        """
        if not authorization:
            return None

        if authorization.startswith("actor:"):
            return authorization[len("actor:"):]

        token = authorization.removeprefix("Bearer ").strip()
        try:
            payload = jwt.decode(token, self._secret, algorithms=["HS256"])
            return payload.get("sub")
        except jwt.PyJWTError:
            return None
