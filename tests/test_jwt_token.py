"""Contract 2 — execution token issuance tests."""
import time
from uuid import uuid4

import jwt
import pytest

from xnch.auth.keys import load_or_generate_keypair
from xnch.auth.token import TokenSigner, ExecutionTokenClaims


@pytest.fixture(scope="module")
def keypair(tmp_path_factory):
    return load_or_generate_keypair(tmp_path_factory.mktemp("keys"))


@pytest.fixture
def signer(keypair):
    return TokenSigner(keypair.private_pem)


def _claims() -> ExecutionTokenClaims:
    return ExecutionTokenClaims(
        session_id=uuid4(),
        decision_id=uuid4(),
        trace_id=uuid4(),
        actor_id="operator",
        actor_role="operator",
        action_type="DEPLOY",
        entity_class="ML_MODEL",
        policy_version="v1.0",
        system_state_version="v3",
    )


def test_token_is_rs256(signer, keypair):
    token, _ = signer.issue(_claims())
    header = jwt.get_unverified_header(token)
    assert header["alg"] == "RS256"


def test_token_required_fields(signer, keypair):
    token, _ = signer.issue(_claims())
    payload = jwt.decode(token, keypair.public_pem, algorithms=["RS256"])
    for field in ["iss", "sub", "jti", "iat", "exp",
                  "session_id", "decision_id", "trace_id",
                  "actor_id", "actor_role", "action_type", "entity_class",
                  "policy_version", "system_state_version", "token_ttl_ms",
                  "role", "trust_level"]:
        assert field in payload, f"Missing required field: {field}"


def test_iss_and_sub_literals(signer, keypair):
    token, _ = signer.issue(_claims())
    payload = jwt.decode(token, keypair.public_pem, algorithms=["RS256"])
    assert payload["iss"] == "xnch"
    assert payload["sub"] == "execution_token"


def test_token_ttl_ms_returned(signer):
    _, ttl_ms = signer.issue(_claims())
    assert ttl_ms == 86_400_000


def test_exp_is_iat_plus_86400(signer, keypair):
    token, _ = signer.issue(_claims())
    payload = jwt.decode(token, keypair.public_pem, algorithms=["RS256"])
    assert payload["exp"] == payload["iat"] + 86400


def test_public_key_verifies(keypair):
    signer = TokenSigner(keypair.private_pem)
    token, _ = signer.issue(_claims())
    payload = jwt.decode(token, keypair.public_pem, algorithms=["RS256"])
    assert payload["actor_role"] == "operator"


def test_wrong_key_rejected(tmp_path):
    other = load_or_generate_keypair(tmp_path / "other")
    signer = TokenSigner(other.private_pem)
    token, _ = signer.issue(_claims())

    from xnch.auth.keys import load_or_generate_keypair as lk
    keypair2 = lk(tmp_path / "kp2")
    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(token, keypair2.public_pem, algorithms=["RS256"])
