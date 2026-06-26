"""xnch /system/state endpoint tests."""
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_app_state():
    """Create mock app state for testing."""
    state = MagicMock()
    state.get_state_version = AsyncMock(return_value="v2.0.0")
    state.get_policy_version = AsyncMock(return_value="policy-v1.5.0")
    return state


@pytest.mark.asyncio
async def test_system_state_returns_versions():
    """System state endpoint should return version information."""
    from httpx import AsyncClient, ASGITransport
    from xnch.main import app

    state = MagicMock()
    state.get_state_version = AsyncMock(return_value="v2.0.0")
    state.get_policy_version = AsyncMock(return_value="policy-v1.5.0")
    app.state = state

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/system/state")

    assert response.status_code == 200
    data = response.json()
    assert "system_state_version" in data
    assert "policy_version" in data


@pytest.mark.asyncio
async def test_system_state_returns_correct_versions():
    """System state should return correct version values."""
    from httpx import AsyncClient, ASGITransport
    from xnch.main import app

    state = MagicMock()
    state.get_state_version = AsyncMock(return_value="v3.0.0")
    state.get_policy_version = AsyncMock(return_value="policy-v2.1.0")
    app.state = state

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/system/state")

    data = response.json()
    assert data["system_state_version"] == "v3.0.0"
    assert data["policy_version"] == "policy-v2.1.0"


@pytest.mark.asyncio
async def test_system_state_version_format():
    """System state version should be a string."""
    from httpx import AsyncClient, ASGITransport
    from xnch.main import app

    state = MagicMock()
    state.get_state_version = AsyncMock(return_value="v1.0.0")
    state.get_policy_version = AsyncMock(return_value="v1.0.0")
    app.state = state

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/system/state")

    data = response.json()
    assert isinstance(data["system_state_version"], str)
    assert isinstance(data["policy_version"], str)


@pytest.mark.asyncio
async def test_system_state_both_versions_present():
    """System state should always include both versions."""
    from httpx import AsyncClient, ASGITransport
    from xnch.main import app

    state = MagicMock()
    state.get_state_version = AsyncMock(return_value="test-state")
    state.get_policy_version = AsyncMock(return_value="test-policy")
    app.state = state

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/system/state")

    data = response.json()
    assert "system_state_version" in data
    assert "policy_version" in data