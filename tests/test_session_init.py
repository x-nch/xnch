"""xnch /session/init endpoint tests."""
import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from xnch.main import app as xnch_app
from xnch.auth.token import TokenSigner


@pytest.fixture
def mock_app_state():
    """Create fully mocked app state for session init tests."""
    state = MagicMock()
    
    state.event_log = MagicMock()
    state.event_log.emit = MagicMock()
    
    kv_cache = MagicMock()
    kv_cache.get_session = AsyncMock(return_value=None)
    kv_cache.set_session = AsyncMock(return_value=True)
    kv_cache.check_rate_limit = AsyncMock(return_value=True)
    state.kv_cache = kv_cache
    
    token_verifier = MagicMock()
    token_verifier.verify_bearer = MagicMock(return_value="test-user")
    state.token_verifier = token_verifier
    
    governance = MagicMock()
    mock_actor = MagicMock()
    mock_actor.to_dict = MagicMock(return_value={
        "id": "test-user", 
        "role": "OPERATOR", 
        "capability_set": ["READ", "WRITE"]
    })
    governance.resolve_actor = AsyncMock(return_value=mock_actor)
    state.governance = governance
    
    state.get_state_version = AsyncMock(return_value="v1.0.0")
    state.get_policy_version = AsyncMock(return_value="v1.0.0")
    
    return state


def valid_session_payload():
    """Valid session init request payload."""
    return {
        "auth_token": "Bearer valid-jwt-token",
        "raw_input": "deploy service myapp",
        "input_type": "TEXT",
        "priority": "NORMAL",
    }


@pytest.mark.asyncio
async def test_session_init_requires_auth_token(mock_app_state):
    """Session init should reject missing auth token (422 validation error)."""
    from httpx import AsyncClient, ASGITransport

    xnch_app.state = mock_app_state
    payload = {"raw_input": "test"}

    transport = ASGITransport(app=xnch_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/session/init", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_session_init_requires_raw_input(mock_app_state):
    """Session init should reject missing raw_input (422 validation error)."""
    from httpx import AsyncClient, ASGITransport

    xnch_app.state = mock_app_state
    payload = {"auth_token": "Bearer test"}

    transport = ASGITransport(app=xnch_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/session/init", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_session_init_invalid_auth_token_rejected(mock_app_state):
    """Session init should reject invalid auth token (401)."""
    from httpx import AsyncClient, ASGITransport

    mock_app_state.token_verifier.verify_bearer = MagicMock(return_value=None)
    xnch_app.state = mock_app_state
    payload = valid_session_payload()

    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"status": "EXECUTING"})
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_client_instance

        transport = ASGITransport(app=xnch_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/session/init", json=payload)

    assert response.status_code == 401
    assert "Invalid auth token" in response.json().get("detail", "")


@pytest.mark.asyncio
async def test_session_init_unknown_actor_rejected(mock_app_state):
    """Session init should reject unknown actor (401)."""
    from httpx import AsyncClient, ASGITransport

    mock_app_state.governance.resolve_actor = AsyncMock(return_value=None)
    xnch_app.state = mock_app_state
    payload = valid_session_payload()

    with patch("httpx.AsyncClient"):
        transport = ASGITransport(app=xnch_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/session/init", json=payload)

    assert response.status_code == 401
    assert "Unknown actor" in response.json().get("detail", "")


@pytest.mark.asyncio
async def test_session_init_rate_limited(mock_app_state):
    """Session init should return 429 when rate limited."""
    from httpx import AsyncClient, ASGITransport

    mock_app_state.kv_cache.check_rate_limit = AsyncMock(return_value=False)
    xnch_app.state = mock_app_state
    payload = valid_session_payload()

    transport = ASGITransport(app=xnch_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/session/init", json=payload)

    assert response.status_code == 429
    assert "Rate limit" in response.json().get("detail", "")


@pytest.mark.asyncio
async def test_session_init_success(mock_app_state):
    """Session init should succeed with valid token and return nexi response."""
    from httpx import AsyncClient, ASGITransport

    xnch_app.state = mock_app_state
    payload = valid_session_payload()

    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={
            "status": "EXECUTING",
            "session_id": str(uuid4()),
            "decision_id": str(uuid4()),
        })
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_client_instance

        transport = ASGITransport(app=xnch_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/session/init", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "status" in data


@pytest.mark.asyncio
async def test_session_init_accepts_priority(mock_app_state):
    """Session init should accept CRITICAL priority."""
    from httpx import AsyncClient, ASGITransport

    xnch_app.state = mock_app_state
    payload = valid_session_payload()
    payload["priority"] = "CRITICAL"

    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"status": "EXECUTING"})
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_client_instance

        transport = ASGITransport(app=xnch_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/session/init", json=payload)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_session_init_accepts_input_type(mock_app_state):
    """Session init should accept VOICE input_type."""
    from httpx import AsyncClient, ASGITransport

    xnch_app.state = mock_app_state
    payload = valid_session_payload()
    payload["input_type"] = "VOICE"

    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"status": "EXECUTING"})
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_client_instance

        transport = ASGITransport(app=xnch_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/session/init", json=payload)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_session_init_dedup_returns_cached(mock_app_state):
    """Session init should return cached response for duplicate idempotency_key."""
    from httpx import AsyncClient, ASGITransport

    cached_response = {"status": "EXECUTING", "session_id": str(uuid4())}
    mock_app_state.kv_cache.get_session = AsyncMock(return_value=cached_response)
    xnch_app.state = mock_app_state

    payload = valid_session_payload()
    payload["idempotency_key"] = str(uuid4())

    transport = ASGITransport(app=xnch_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/session/init", json=payload)

    assert response.status_code == 200
    assert response.json() == cached_response


@pytest.mark.asyncio
async def test_session_init_nexi_unavailable_returns_502(mock_app_state):
    """Session init should return 502 when Nexi is unavailable."""
    from httpx import AsyncClient, ASGITransport

    xnch_app.state = mock_app_state
    payload = valid_session_payload()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.side_effect = Exception("Connection refused")

        transport = ASGITransport(app=xnch_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/session/init", json=payload)

    assert response.status_code == 502
    assert "Nexi unavailable" in response.json().get("detail", "")