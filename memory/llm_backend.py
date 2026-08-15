"""In-process LLM backend using llama-cpp-python (llama.cpp).

Replaces the Ollama daemon for the graph extractor. The model loads once
and stays resident in memory — subsequent extraction calls are near-zero
startup overhead (no HTTP loop, no daemon process).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

_MODEL_PATH: Path | None = None
_MODEL_LOADED: Any = None
_BACKEND: str = "ollama"  # "llama_cpp" | "ollama"


def _resolve_model_path() -> Path:
    """Return the path to the GGUF model file, downloading it if needed."""
    global _MODEL_PATH
    if _MODEL_PATH is not None:
        return _MODEL_PATH

    model_dir = settings.base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check environment override first
    env_model = settings.model_dump().get("graph_extractor_model", "")
    if env_model and env_model.startswith("llama_cpp/"):
        filename = env_model.split("/", 1)[1]
        candidate = model_dir / filename
        if candidate.exists():
            _MODEL_PATH = candidate
            return _MODEL_PATH

    # Auto-detect the best small model available (preferred first)
    candidates = [
        "qwen2.5-0.5b-instruct-q4_0.gguf",
        "qwen2.5-1.5b-instruct-q4_0.gguf",
        "smollm2-1.7b-instruct-q4_0.gguf",
    ]
    for name in candidates:
        candidate = model_dir / name
        if candidate.exists():
            _MODEL_PATH = candidate
            logger.info("Using local model: %s", candidate)
            return _MODEL_PATH

    # Fall back to any .gguf file already downloaded
    gguf_files = list(model_dir.glob("*.gguf"))
    if gguf_files:
        _MODEL_PATH = gguf_files[0]
        return _MODEL_PATH

    raise FileNotFoundError(
        "No GGUF model found in models/ directory. "
        "Run: bash scripts/setup-llm.sh <node-a-ip>"
    )


def _load_model() -> Any:
    """Load the GGUF model into memory (singleton)."""
    global _MODEL_LOADED
    if _MODEL_LOADED is not None:
        return _MODEL_LOADED

    model_path = _resolve_model_path()
    n_threads = min(max(1, __import__("os").cpu_count() // 2), 4)

    from llama_cpp import Llama

    logger.info(
        "Loading %s (threads=%d)", model_path, n_threads
    )
    _MODEL_LOADED = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_batch=512,
        n_threads=n_threads,
        verbose=False,
    )
    return _MODEL_LOADED


def _clean_json(content: str) -> str:
    """Strip markdown code fences from LLM output."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    return content.strip()


def chat_completion(messages: list[dict[str, str]], **kwargs: Any) -> str:
    """Run a single chat completion against the loaded model.

    Returns the raw string content of the first choice.
    """
    model = _load_model()

    response = model.create_chat_completion(
        messages=messages,
        temperature=kwargs.get("temperature", 0.1),
        max_tokens=kwargs.get("max_tokens", 1024),
        stream=False,
    )

    return response["choices"][0]["message"]["content"]


def close() -> None:
    """Release model from memory."""
    global _MODEL_LOADED
    if _MODEL_LOADED is not None:
        del _MODEL_LOADED
        _MODEL_LOADED = None