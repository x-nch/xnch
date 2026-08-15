"""Tests for graph_extractor backend selection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import xnch.memory.graph_extractor as gmod


TRIPLE_A = {
    "subject": {"id": "a", "name": "A", "type": "svc"},
    "relation": "uses",
    "object": {"id": "b", "name": "B", "type": "svc"},
}


class TestBackendSelection:
    """Test which backend is selected for extraction."""

    async def test_uses_llama_cpp_when_gguf_exists_no_ollama(self):
        """When GGUF exists and Ollama is unreachable, use llama_cpp."""
        with patch.object(gmod, "_use_llama_cpp", return_value=True), \
             patch.object(gmod, "_extract_llama_cpp", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = [TRIPLE_A]
            result = await gmod._extract_triples("test text")
            assert result == [TRIPLE_A]
            mock_extract.assert_awaited_once_with("test text")

    async def test_returns_empty_list_on_llama_cpp_failure(self):
        """If the llama.cpp backend raises, _extract_llama_cpp propagates the error."""
        from xnch.memory import llm_backend

        with patch.object(llm_backend, "chat_completion", side_effect=RuntimeError("model load failed")):
            with pytest.raises(RuntimeError):
                await gmod._extract_llama_cpp("test")

    async def test_fallback_to_litellm_when_no_gguf(self):
        """If no GGUF exists, falls back to LiteLLM."""
        with patch.object(gmod, "_use_llama_cpp", return_value=False), \
             patch.object(gmod, "_extract_litellm", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = [TRIPLE_A]
            result = await gmod._extract_triples("test")
            assert result == [TRIPLE_A]
            mock_extract.assert_awaited_once_with("test")

    async def test_litellm_failure_returns_empty(self):
        """If LiteLLM raises, _extract_litellm propagates and _extract_triples raises."""
        with patch.object(gmod, "_use_llama_cpp", return_value=False), \
             patch.object(gmod, "_extract_litellm", new_callable=AsyncMock) as mock_extract:
            mock_extract.side_effect = RuntimeError("litellm not reachable")
            with pytest.raises(RuntimeError):
                await gmod._extract_triples("test")


class TestExtractionPrompt:
    """The prompt template must survive .format() despite JSON braces."""

    def test_prompt_formats_without_key_error(self):
        formatted = gmod._EXTRACTION_PROMPT.format(raw_text="episode text")
        assert "Episode:\nepisode text" in formatted
        assert '"subject": {"id": str, "name": str, "type": str}' in formatted
        assert '{"id": str' in formatted

    def test_prompt_escaped_braces_render_single(self):
        formatted = gmod._EXTRACTION_PROMPT.format(raw_text="episode text")
        assert "{{" not in formatted
        assert "}}" not in formatted


class TestLiteLLMProxyModel:
    """Bare model names get an openai/ prefix when routed via the proxy."""

    async def test_does_not_prefix_provider_qualified_model(self):
        with patch("xnch.memory.graph_extractor.litellm.acompletion", new_callable=AsyncMock) as mock, \
             patch("xnch.config.settings") as mock_settings:
            mock_settings.graph_extractor_model = "ollama/phi3:mini"
            mock_settings.litellm_proxy_url = "http://localhost:4000"
            mock.return_value.choices = [MagicMock(message=MagicMock(content="[]"))]
            await gmod._extract_litellm("text")
            assert mock.await_args.kwargs["model"] == "ollama/phi3:mini"

    async def test_prefixes_bare_model_with_openai(self):
        with patch("xnch.memory.graph_extractor.litellm.acompletion", new_callable=AsyncMock) as mock, \
             patch("xnch.config.settings") as mock_settings:
            mock_settings.graph_extractor_model = "qwen2.5-vl-7b"
            mock_settings.litellm_proxy_url = "http://localhost:4000"
            mock.return_value.choices = [MagicMock(message=MagicMock(content="[]"))]
            await gmod._extract_litellm("text")
            assert mock.await_args.kwargs["model"] == "openai/qwen2.5-vl-7b"


class TestUseLlamaCpp:
    """Test the _use_llama_cpp backend-selection helper.

    Selection is purely config-driven (XNCH_GRAPH_EXTRACTOR_MODEL prefixed
    with ``llama_cpp/``); GGUF file presence alone must NOT switch backends.
    """

    def _settings(self, model: str) -> MagicMock:
        mock = MagicMock()
        mock.graph_extractor_model = model
        return mock

    def test_true_when_config_prefix_llama_cpp(self):
        with patch("xnch.memory.graph_extractor.settings", self._settings("llama_cpp/qwen2.5-0.5b.gguf")):
            assert gmod._use_llama_cpp() is True

    def test_false_when_remote_model_configured(self):
        with patch("xnch.memory.graph_extractor.settings", self._settings("ornith")):
            assert gmod._use_llama_cpp() is False

    def test_false_when_empty(self):
        with patch("xnch.memory.graph_extractor.settings", self._settings("")):
            assert gmod._use_llama_cpp() is False


class TestNormalizeTriple:
    """Test _normalize_triple coercion for small-model output."""

    def test_passthrough_dict_entities(self):
        t = {"subject": {"id": "a", "name": "A", "type": "svc"}, "relation": "uses", "object": {"id": "b", "name": "B", "type": "svc"}}
        assert gmod._normalize_triple(t) == t

    def test_coerces_string_subject_and_object(self):
        t = {"subject": "ck-san", "relation": "pivots_to", "object": {"id": "FDE", "name": "FDE", "type": "role"}}
        assert gmod._normalize_triple(t) == {
            "subject": {"id": "ck-san", "name": "ck-san", "type": "entity"},
            "relation": "pivots_to",
            "object": {"id": "FDE", "name": "FDE", "type": "role"},
        }

    def test_fills_missing_name_from_id(self):
        t = {"subject": {"id": "svc-1"}, "relation": "deploys", "object": "gate7"}
        assert gmod._normalize_triple(t) == {
            "subject": {"id": "svc-1", "name": "svc-1", "type": "entity"},
            "relation": "deploys",
            "object": {"id": "gate7", "name": "gate7", "type": "entity"},
        }

    def test_returns_none_for_non_dict(self):
        assert gmod._normalize_triple("not-a-dict") is None
        assert gmod._normalize_triple(None) is None

    def test_returns_none_for_missing_fields(self):
        assert gmod._normalize_triple({"subject": "a"}) is None
        assert gmod._normalize_triple({"subject": "a", "relation": "r"}) is None

    def test_returns_none_for_empty_entity(self):
        t = {"subject": "", "relation": "r", "object": {"id": "b", "name": "B", "type": "svc"}}
        assert gmod._normalize_triple(t) is None
