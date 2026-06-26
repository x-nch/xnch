"""xnch /health endpoint tests."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def mock_app_state():
    """Create mock app state for testing."""
    state = MagicMock()
    
    # Mock KV cache
    kv_cache = AsyncMock()
    kv_cache.ping = AsyncMock(return_value=True)
    state.kv_cache = kv_cache
    
    # Mock version functions
    state.get_state_version = AsyncMock(return_value="v1.0.0")
    state.get_policy_version = AsyncMock(return_value="v1.0.0")
    
    return state


@pytest.mark.asyncio
async def test_health_returns_ok_status(mock_app_state):
    """Health endpoint should return ok status when Redis is available."""
    from httpx import AsyncClient, ASGITransport
    from xnch.main import app

    # Override app state
    app.state = mock_app_state

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_health_returns_degraded_when_redis_unavailable(mock_app_state):
    """Health endpoint should return degraded when Redis is unavailable."""
    from httpx import AsyncClient, ASGITransport
    from xnch.main import app

    # Override app state before modifying mocks
    app.state = mock_app_state

    # Make Redis unavailable
    mock_app_state.kv_cache.ping = AsyncMock(return_value=False)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["redis"] == "unavailable"


@pytest.mark.asyncio
async def test_health_includes_versions(mock_app_state):
    """Health endpoint should include state and app versions."""
    from httpx import AsyncClient, ASGITransport
    from xnch.main import app

    app.state = mock_app_state

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    data = response.json()
    assert "version" in data
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_health_includes_state_version(mock_app_state):
    """Health endpoint should include state_version."""
    from httpx import AsyncClient, ASGITransport
    from xnch.main import app

    app.state = mock_app_state

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    data = response.json()
    assert "state_version" in data


@pytest.mark.asyncio
async def test_health_redis_check(mock_app_state):
    """Health endpoint should check Redis connectivity."""
    from httpx import AsyncClient, ASGITransport
    from xnch.main import app

    app.state = mock_app_state

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    data = response.json()
    assert "redis" in data
    mock_app_state.kv_cache.ping.assert_called_once()