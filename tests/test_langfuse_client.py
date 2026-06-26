"""Unit tests for LangfuseClient (mock HTTP, no real credentials)."""
import pytest
from unittest.mock import patch, MagicMock

from unittest.mock import AsyncMock

from xnch.observability.langfuse_client import LangfuseClient, trace_llm_call, get_client


class TestLangfuseClientInit:
    def test_uses_provided_keys(self):
        client = LangfuseClient(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://example.com",
        )
        assert client._public_key == "pk-test"
        assert client._secret_key == "sk-test"
        assert client._host == "https://example.com"

    @pytest.mark.asyncio
    async def test_returns_none_when_keys_missing(self):
        client = LangfuseClient(public_key="", secret_key="")
        result = await client.trace_llm_call(
            prompt="hello",
            response="world",
            model="gpt-4",
            latency_ms=100,
            tokens_used=10,
        )
        assert result is None


class TestTraceLlmCall:
    @pytest.mark.asyncio
    async def test_sends_ingestion_request(self):
        client = LangfuseClient(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://example.com",
        )
        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await client.trace_llm_call(
                prompt="What is 2+2?",
                response="4",
                model="gpt-4",
                latency_ms=150,
                tokens_used=25,
                trace_id="trace-123",
            )

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "/api/public/ingestion"
        body = kwargs["json"]["batch"][0]["body"]
        assert body["model"] == "gpt-4"
        assert body["latency"] == 150
        assert body["traceId"] == "trace-123"
        assert result is not None

    @pytest.mark.asyncio
    async def test_includes_usage_stats(self):
        client = LangfuseClient(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://example.com",
        )
        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            await client.trace_llm_call(
                prompt="hello world foo",
                response="bar baz",
                model="claude-3",
                latency_ms=200,
                tokens_used=30,
            )

            body = mock_post.call_args[1]["json"]["batch"][0]["body"]
            usage = body["usage"]
            assert usage["input"] == 3
            assert usage["output"] == 2
            assert usage["total"] == 5

    @pytest.mark.asyncio
    async def test_handles_http_error_gracefully(self):
        client = LangfuseClient(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://example.com",
        )
        with patch.object(client._client, "post") as mock_post:
            mock_post.side_effect = Exception("connection error")

            result = await client.trace_llm_call(
                prompt="test",
                response="test",
                model="gpt-4",
                latency_ms=50,
                tokens_used=5,
            )
            assert result is None


class TestModuleFunction:
    @pytest.mark.asyncio
    async def test_trace_llm_call_module_function(self):
        with patch("xnch.observability.langfuse_client.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.trace_llm_call = AsyncMock(return_value={"id": "abc"})
            mock_get.return_value = mock_client

            result = await trace_llm_call(
                prompt="hello",
                response="world",
                model="gpt-4",
                latency_ms=100,
                tokens_used=10,
            )
            assert result == {"id": "abc"}
            mock_client.trace_llm_call.assert_called_once_with(
                prompt="hello",
                response="world",
                model="gpt-4",
                latency_ms=100,
                tokens_used=10,
                trace_id="",
            )

    def test_get_client_singleton(self):
        with patch("xnch.observability.langfuse_client.LangfuseClient") as MockClass:
            first = get_client()
            second = get_client()
            assert first is second
