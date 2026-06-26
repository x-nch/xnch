"""xnch Auth verification tests."""
import time
import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
import jwt as pyjwt

from xnch.auth.token import TokenVerifier
from xnch.config import settings


@pytest.fixture
def secret():
    return settings.auth_secret


@pytest.fixture
def verifier(secret):
    return TokenVerifier(secret)


class TestTokenVerifier:
    """Token verification edge cases."""

    def test_valid_bearer_token_returns_subject(self, verifier, secret):
        """Valid HS256 token should return the subject (actor_id)."""
        payload = {"sub": "operator-1", "iss": "xnch"}
        token = pyjwt.encode(payload, secret, algorithm="HS256")
        result = verifier.verify_bearer(f"Bearer {token}")
        assert result == "operator-1"

    def test_valid_actor_reference(self, verifier):
        """Plain 'actor:' format should return the actor ID."""
        result = verifier.verify_bearer("actor:admin-001")
        assert result == "admin-001"

    def test_missing_authorization_returns_none(self, verifier):
        """Empty/None authorization should return None."""
        assert verifier.verify_bearer("") is None
        assert verifier.verify_bearer(None) is None

    def test_invalid_token_returns_none(self, verifier, secret):
        """Malformed token should return None."""
        result = verifier.verify_bearer("Bearer invalid.token.here")
        assert result is None

    def test_wrong_signature_returns_none(self, verifier):
        """Token signed with wrong secret should return None."""
        payload = {"sub": "actor-1", "iss": "xnch"}
        token = pyjwt.encode(payload, "wrong-secret-32bytes-long!!!!", algorithm="HS256")
        result = verifier.verify_bearer(f"Bearer {token}")
        assert result is None

    def test_expired_token_returns_none(self, verifier, secret):
        """Expired token should return None."""
        payload = {
            "sub": "actor-1",
            "iss": "xnch",
            "exp": int(time.time()) - 3600,  # 1 hour ago
        }
        token = pyjwt.encode(payload, secret, algorithm="HS256")
        result = verifier.verify_bearer(f"Bearer {token}")
        assert result is None

    def test_missing_subject_returns_none(self, verifier, secret):
        """Token without 'sub' claim should return None."""
        payload = {"iss": "xnch"}
        token = pyjwt.encode(payload, secret, algorithm="HS256")
        result = verifier.verify_bearer(f"Bearer {token}")
        assert result is None

    def test_bearer_prefix_stripped(self, verifier, secret):
        """Bearer prefix should be stripped correctly."""
        payload = {"sub": "actor-1", "iss": "xnch"}
        token = pyjwt.encode(payload, secret, algorithm="HS256")
        
        # With Bearer prefix
        result = verifier.verify_bearer(f"Bearer {token}")
        assert result == "actor-1"
        
        # Without Bearer prefix (should still work with HS256 format)
        result2 = verifier.verify_bearer(token)
        assert result2 == "actor-1"

    def test_whitespace_in_bearer_header(self, verifier, secret):
        """Extra whitespace in Bearer header should be handled."""
        payload = {"sub": "actor-1", "iss": "xnch"}
        token = pyjwt.encode(payload, secret, algorithm="HS256")
        result = verifier.verify_bearer(f"Bearer   {token}   ")
        assert result == "actor-1"


class TestJtiReplayProtection:
    """JTI replay protection tests."""

    def test_first_use_returns_true(self):
        """New JTI should be consumed successfully."""
        from xnch.auth.token import _JtiSeenSet
        jti_set = _JtiSeenSet()
        result = jti_set.consume("unique-jti-1", time.time() + 60)
        assert result is True

    def test_replay_returns_false(self):
        """Replayed JTI should return False."""
        from xnch.auth.token import _JtiSeenSet
        jti_set = _JtiSeenSet()
        jti_set.consume("replayed-jti", time.time() + 60)
        result = jti_set.consume("replayed-jti", time.time() + 60)
        assert result is False

    def test_expired_jti_can_be_reused(self):
        """Expired JTI should be evictable and reusable."""
        from xnch.auth.token import _JtiSeenSet
        jti_set = _JtiSeenSet()
        
        # Use JTI that's already expired
        jti_set.consume("stale-jti", time.time() - 1)
        
        # Should be able to reuse it
        result = jti_set.consume("stale-jti", time.time() + 60)
        assert result is True

    def test_replay_returns_false(self, verifier):
        """Replayed JTI should return False."""
        from xnch.auth.token import _JtiSeenSet
        jti_set = _JtiSeenSet()
        jti_set.consume("replayed-jti", time.time() + 60)
        result = jti_set.consume("replayed-jti", time.time() + 60)
        assert result is False

    def test_expired_jti_can_be_reused(self, verifier):
        """Expired JTI should be evictable and reusable."""
        from xnch.auth.token import _JtiSeenSet
        jti_set = _JtiSeenSet()
        
        # Use JTI that's already expired
        jti_set.consume("stale-jti", time.time() - 1)
        
        # Should be able to reuse it
        result = jti_set.consume("stale-jti", time.time() + 60)
        assert result is True