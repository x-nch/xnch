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


class _FakeResp:
    status_code = 200
    text = ""

    def __init__(self, content: str):
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeProxyClient:
    url = ""
    payload = {}
    headers = {}

    def __init__(self, content: str = "[]", status_code: int = 200):
        self._content = content
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        type(self).url = url
        type(self).payload = json or {}
        type(self).headers = headers or {}
        resp = _FakeResp(self._content)
        resp.status_code = self.status_code
        return resp


def _proxy(monkeypatch, content="[]", status_code=200, model="ornith", hint="", key="sk-test"):
    monkeypatch.setattr("xnch.config.settings.opencode_go_api_key", key)
    monkeypatch.setattr("xnch.config.settings.opencode_go_api_url", "https://opencode.ai/zen/go/v1")
    monkeypatch.setattr(gmod.httpx, "AsyncClient", lambda **kw: _FakeProxyClient(content, status_code))
    monkeypatch.setattr("xnch.config.settings.graph_extractor_model", model)
    monkeypatch.setattr("xnch.config.settings.graph_extractor_provider_hint", hint)


class TestOpenCodeGoModel:
    """Extraction posts straight to the OpenCode Go API endpoint with the id verbatim."""

    async def test_posts_to_openai_compatible_endpoint_with_verbatim_model(self, monkeypatch):
        _proxy(monkeypatch, model="ornith")
        assert await gmod._extract_litellm("text") == []
        assert _FakeProxyClient.url.endswith("/v1/chat/completions")
        assert _FakeProxyClient.payload["model"] == "ornith"

    async def test_sends_api_key_as_bearer_header(self, monkeypatch):
        _proxy(monkeypatch, key="sk-master")
        await gmod._extract_litellm("text")
        assert _FakeProxyClient.headers["Authorization"] == "Bearer sk-master"

    async def test_provider_hint_opt_in_prefixes_model(self, monkeypatch):
        _proxy(monkeypatch, model="ornith", hint="openai")
        await gmod._extract_litellm("text")
        assert _FakeProxyClient.payload["model"] == "openai/ornith"

    async def test_truncates_long_episode(self, monkeypatch):
        _proxy(monkeypatch)
        sent = await gmod._extract_litellm("x" * 12000)
        user_msg = _FakeProxyClient.payload["messages"][1]["content"]
        assert len(user_msg) < 6500
        assert "[truncated]" in user_msg

    async def test_recovers_json_array_from_prose(self, monkeypatch):
        content = 'Here is the result:\n[{"subject": {"id": "a", "name": "A", "type": "svc"}, "relation": "uses", "object": {"id": "b", "name": "B", "type": "svc"}}]\nDone!'
        _proxy(monkeypatch, content=content)
        result = await gmod._extract_litellm("text")
        assert len(result) == 1
        assert result[0]["relation"] == "uses"

    async def test_unparseable_content_treated_as_no_triples(self, monkeypatch):
        _proxy(monkeypatch, content="I'm sorry, I cannot extract triples from this text.")
        assert await gmod._extract_litellm("text") == []

    async def test_http_error_raises_for_skip_retry(self, monkeypatch):
        _proxy(monkeypatch, status_code=400)
        with pytest.raises(RuntimeError, match="400"):
            await gmod._extract_litellm("text")


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


class TestParseTriplesJson:
    """Robust extraction: reasoning models emit draft arrays before the final one."""

    ARR = '[{"subject": {"id": "a", "name": "A", "type": "svc"}, "relation": "uses", "object": {"id": "b", "name": "B", "type": "svc"}}]'
    DRAFT = '[{"subject": {"id": "x", "name": "X", "type": "?"}, "relation": "maybe", "object": {"id": "y", "name": "Y", "type": "?"}}]'

    def _cot(self) -> str:
        return (
            "Here's a thinking process:\n"
            f"draft: {self.DRAFT}\n"
            "hmm, relation should be uses not maybe...\n"
            f"Final answer:\n{self.ARR}\n"
            "This extracts the service dependency."
        )

    def test_cot_draft_then_final_array_wins(self):
        result = gmod._parse_triples_json(self._cot())
        assert len(result) == 1
        assert result[0]["relation"] == "uses"

    def test_single_array_in_prose(self):
        result = gmod._parse_triples_json(f"Sure:\n{self.ARR}")
        assert result[0]["relation"] == "uses"

    def test_plain_array_passthrough(self):
        assert gmod._parse_triples_json(self.ARR)[0]["relation"] == "uses"

    def test_object_wrapper_not_returned_as_triples(self):
        assert gmod._parse_triples_json('{"triples": [1,2]}') == []

    def test_no_arrays_returns_empty(self):
        assert gmod._parse_triples_json("I cannot help with that.") == []

    def test_citation_noise_does_not_beat_real_array(self):
        result = gmod._parse_triples_json(f"{self.ARR}\nSee refs [1] and [2] for details.")
        assert len(result) == 1
        assert result[0]["relation"] == "uses"

    def test_extra_data_case_from_live_failure(self):
        content = (
            "Here's a thinking process:\n"
            f"{self.DRAFT}\n"
            f"{self.ARR}\n"
            "trailing prose"
        )
        result = gmod._parse_triples_json(content)
        assert result[0]["relation"] == "uses"


