"""Tests for the in-process LLM backend (llama-cpp-python)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xnch.memory import llm_backend


@pytest.fixture(autouse=True)
def reset_backend_cache():
    """Reset module-level model cache between tests."""
    llm_backend._MODEL_PATH = None
    llm_backend._MODEL_LOADED = None
    yield
    llm_backend._MODEL_PATH = None
    llm_backend._MODEL_LOADED = None


def _patch_llama_cpp(mock_llama: MagicMock) -> object:
    """Build a fake llama_cpp module so tests don't need the real package."""
    fake = types.ModuleType("llama_cpp")
    fake.Llama = MagicMock(return_value=mock_llama)
    return fake


def _model_in(tmp_path: Path, name: str = "test.gguf") -> Path:
    """Create a fake model inside the settings model_dir."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / name
    model.write_bytes(b"\x00")
    return model


def _mock_settings(tmp_path: Path, model_name: str | None = None) -> MagicMock:
    mock = MagicMock()
    mock.base_dir = tmp_path
    mock.model_dump.return_value = (
        {"graph_extractor_model": f"llama_cpp/{model_name}"}
        if model_name
        else {}
    )
    return mock


class TestCleanJson:
    """Test the JSON cleaning function."""

    def test_strips_markdown_code_fences(self):
        raw = '```json\n[{"a": 1}]\n```'
        assert llm_backend._clean_json(raw) == '[{"a": 1}]'

    def test_strips_plain_code_fences(self):
        raw = '```something\n[{"a": 1}]\n```'
        assert llm_backend._clean_json(raw) == '[{"a": 1}]'

    def test_passthrough_clean_json(self):
        raw = '[{"a": 1}]'
        assert llm_backend._clean_json(raw) == '[{"a": 1}]'

    def test_strips_leading_trailing_whitespace(self):
        raw = '\n  [1, 2]  \n'
        assert llm_backend._clean_json(raw) == '[1, 2]'


class TestResolveModelPath:
    """Test model path resolution with mocked filesystem."""

    def test_env_override_takes_precedence(self, tmp_path):
        fake = _model_in(tmp_path, "foo.gguf")
        with patch("xnch.memory.llm_backend.settings", _mock_settings(tmp_path, "foo.gguf")):
            path = llm_backend._resolve_model_path()
            assert path == fake

    def test_finds_gguf_in_model_dir(self, tmp_path):
        gguf = _model_in(tmp_path, "qwen2.5-0.5b-instruct-q4_0.gguf")
        with patch("xnch.memory.llm_backend.settings", _mock_settings(tmp_path)):
            path = llm_backend._resolve_model_path()
            assert path == gguf

    def test_auto_detect_qwen25_before_smollm2(self, tmp_path):
        qwen = _model_in(tmp_path, "qwen2.5-0.5b-instruct-q4_0.gguf")
        _model_in(tmp_path, "smollm2-1.7b-instruct-q4_0.gguf")
        with patch("xnch.memory.llm_backend.settings", _mock_settings(tmp_path)):
            path = llm_backend._resolve_model_path()
            assert path == qwen

    def test_raises_when_no_model_found(self, tmp_path):
        with patch("xnch.memory.llm_backend.settings", _mock_settings(tmp_path)):
            with pytest.raises(FileNotFoundError, match="No GGUF model"):
                llm_backend._resolve_model_path()

    def test_caches_result(self, tmp_path):
        gguf = _model_in(tmp_path, "test.gguf")
        with patch("xnch.memory.llm_backend.settings", _mock_settings(tmp_path)):
            path1 = llm_backend._resolve_model_path()
            path2 = llm_backend._resolve_model_path()
            assert path1 is path2
            assert path1 == gguf


class TestLoadModel:
    """Test model loading with a mocked llama_cpp.Llama."""

    def test_loads_model_once(self, tmp_path):
        _model_in(tmp_path)
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "[{\"a\": 1}]"}}]
        }

        with patch("xnch.memory.llm_backend.settings", _mock_settings(tmp_path)), \
             patch.dict(sys.modules, {"llama_cpp": _patch_llama_cpp(mock_llama)}), \
             patch("xnch.memory.llm_backend._load_model", side_effect=llm_backend._load_model) as _:
            result1 = llm_backend._load_model()
            result2 = llm_backend._load_model()
            assert result1 is result2

    def test_load_caps_threads(self, tmp_path):
        _model_in(tmp_path)
        mock_llama = MagicMock()
        fake_module = _patch_llama_cpp(mock_llama)

        with patch("xnch.memory.llm_backend.settings", _mock_settings(tmp_path)), \
             patch.dict(sys.modules, {"llama_cpp": fake_module}):
            llm_backend._load_model()
            call = fake_module.Llama.call_args
            assert call.kwargs.get("n_threads") == min(max(1, __import__("os").cpu_count() // 2), 4)


class TestChatCompletion:
    """Test the chat_completion function."""

    def test_returns_content_string(self, tmp_path):
        _model_in(tmp_path)
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello world"}}]
        }

        with patch("xnch.memory.llm_backend.settings", _mock_settings(tmp_path)), \
             patch.dict(sys.modules, {"llama_cpp": _patch_llama_cpp(mock_llama)}):
            result = llm_backend.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.1,
                max_tokens=100,
            )
            assert result == "Hello world"

    def test_returns_raw_content_with_fences(self, tmp_path):
        """chat_completion returns the raw response; JSON cleaning is the
        caller's responsibility (graph_extractor applies _clean_json)."""
        _model_in(tmp_path)
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "```json\n[{\"a\":1}]\n```"}}]
        }

        with patch("xnch.memory.llm_backend.settings", _mock_settings(tmp_path)), \
             patch.dict(sys.modules, {"llama_cpp": _patch_llama_cpp(mock_llama)}):
            result = llm_backend.chat_completion(
                messages=[{"role": "user", "content": "extract"}]
            )
            assert result == '```json\n[{"a":1}]\n```'

    def test_passes_temperature_and_max_tokens(self, tmp_path):
        _model_in(tmp_path)
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        with patch("xnch.memory.llm_backend.settings", _mock_settings(tmp_path)), \
             patch.dict(sys.modules, {"llama_cpp": _patch_llama_cpp(mock_llama)}):
            llm_backend.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                max_tokens=2048,
            )
            call_kwargs = mock_llama.create_chat_completion.call_args.kwargs
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 2048

    def test_default_parameters(self, tmp_path):
        _model_in(tmp_path)
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        with patch("xnch.memory.llm_backend.settings", _mock_settings(tmp_path)), \
             patch.dict(sys.modules, {"llama_cpp": _patch_llama_cpp(mock_llama)}):
            llm_backend.chat_completion(
                messages=[{"role": "user", "content": "test"}]
            )
            call_kwargs = mock_llama.create_chat_completion.call_args.kwargs
            assert call_kwargs["temperature"] == 0.1
            assert call_kwargs["max_tokens"] == 1024


class TestClose:
    """Test the close function."""

    def test_clears_loaded_model(self, tmp_path):
        mock_llama = MagicMock()
        llm_backend._MODEL_LOADED = mock_llama
        llm_backend.close()
        assert llm_backend._MODEL_LOADED is None
