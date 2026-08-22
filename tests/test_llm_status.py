"""xnch /system/llm-status endpoint and vLLM probe tests."""
import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def llm_settings(monkeypatch):
    """Point the LLM probe at a test URL with tight timeout."""
    from xnch.config import settings

    monkeypatch.setattr(settings, "llm_status_url", "http://vllm.test:8082/health")
    monkeypatch.setattr(settings, "llm_model_id", "ornith-test-35b")
    monkeypatch.setattr(settings, "llm_probe_timeout_s", 0.5)


@pytest.mark.asyncio
async def test_probe_available_when_vllm_healthy(llm_settings, monkeypatch):
    """Probe returning HTTP 200 should report available with latency."""
    from xnch.main import _probe_vllm

    monkeypatch.setattr(
        "httpx.AsyncClient.get",
        AsyncMock(return_value=MagicMock(status_code=200)),
    )

    available, latency_ms = await _probe_vllm()

    assert available is True
    assert isinstance(latency_ms, int)


@pytest.mark.asyncio
async def test_probe_unavailable_on_connect_error(llm_settings, monkeypatch):
    """Connection failure should report unavailable without raising."""
    from xnch.main import _probe_vllm

    monkeypatch.setattr(
        "httpx.AsyncClient.get",
        AsyncMock(side_effect=httpx.ConnectError("refused")),
    )

    available, latency_ms = await _probe_vllm()

    assert available is False
    assert latency_ms is None


@pytest.mark.asyncio
async def test_probe_unavailable_on_error_status(llm_settings, monkeypatch):
    """Non-200 probe response should report unavailable."""
    from xnch.main import _probe_vllm

    monkeypatch.setattr(
        "httpx.AsyncClient.get",
        AsyncMock(return_value=MagicMock(status_code=503)),
    )

    available, latency_ms = await _probe_vllm()

    assert available is False
    assert latency_ms is None


@pytest.mark.asyncio
async def test_llm_status_route_shape(llm_settings, monkeypatch):
    """Route should expose availability, model id, and latency."""
    from httpx import ASGITransport, AsyncClient
    from xnch.main import app

    monkeypatch.setattr(
        "xnch.main._probe_vllm", AsyncMock(return_value=(True, 12))
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/system/llm-status")

    assert response.status_code == 200
    data = response.json()
    assert data == {"available": True, "model": "ornith-test-35b", "latency_ms": 12}


@pytest.mark.asyncio
async def test_llm_status_route_reports_down(llm_settings, monkeypatch):
    """Route should surface an unavailable LLM without raising."""
    from httpx import ASGITransport, AsyncClient
    from xnch.main import app

    monkeypatch.setattr(
        "xnch.main._probe_vllm", AsyncMock(return_value=(False, None))
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/system/llm-status")

    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["latency_ms"] is None
