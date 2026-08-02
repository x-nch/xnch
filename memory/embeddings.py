"""Local ONNX MiniLM-L6-v2 embeddings — Layer 2/3 semantic search vector source.

Replaces the embedder that chromadb/agentmemory provided. Uses the same
all-MiniLM-L6-v2 ONNX model with identical mean-pooling + L2 normalization,
so vectors remain compatible with any data produced while chromadb was in use.

The model is bundled with chromadb's onnx_models cache; if absent we download
from the HuggingFace source chromadb uses.
"""

from __future__ import annotations

import importlib
import logging
import shutil
import sys
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DIM = 384
_HF_REPO = "chromadb-onnx/all-MiniLM-L6-v2"
_HF_FILENAME = "onnx.tar.gz"
_DEFAULT_CACHE = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"


def _resolve_model_dir() -> Path:
    model_dir = Path.home() / ".xnch" / "models" / "all-MiniLM-L6-v2"
    onnx_file = model_dir / "onnx" / "model.onnx"
    if onnx_file.exists():
        return model_dir

    default = _DEFAULT_CACHE
    if (default / "onnx" / "model.onnx").exists():
        return default

    _download(model_dir)
    return model_dir


def _download(target: Path) -> None:
    import urllib.request

    target.mkdir(parents=True, exist_ok=True)
    tarball = target / _HF_FILENAME
    url = f"https://huggingface.co/{_HF_REPO}/resolve/main/{_HF_FILENAME}"
    logger.info("Downloading embedding model from %s", url)
    urllib.request.urlretrieve(url, tarball)  # noqa: S310
    with zipfile.ZipFile(tarball) as zf:
        zf.extractall(target)
    tarball.unlink()


class MiniLMEmbedder:
    """Mean-pooled, L2-normalized MiniLM-L6-v2 embeddings via onnxruntime."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self._dir = model_dir or _resolve_model_dir()
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(self._dir / "onnx" / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = importlib.import_module("tokenizers").Tokenizer.from_file(
            str(self._dir / "onnx" / "tokenizer.json")
        )

    def max_tokens(self) -> int:
        return 256

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=1)
        norm[norm == 0] = 1e-12
        return v / norm[:, np.newaxis]

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, _DIM), dtype=np.float32)
        encoded = [self._tokenizer.encode(t) for t in texts]
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.array(
            [np.zeros(len(e.ids), dtype=np.int64) for e in encoded],
            dtype=np.int64,
        )
        (last_hidden,) = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, -1), last_hidden.shape
        )
        pooled = np.sum(last_hidden * mask_expanded, 1) / np.clip(
            mask_expanded.sum(1), a_min=1e-9, a_max=None
        )
        return self._normalize(pooled).astype(np.float32)

    def embed(self, text: str) -> list[float]:
        return self._embed_batch([text])[0].tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._embed_batch(texts)]

    def close(self) -> None:
        try:
            self._session = None  # type: ignore[assignment]
        except Exception:  # pragma: no cover
            pass


@lru_cache(maxsize=1)
def get_embedder() -> MiniLMEmbedder:
    return MiniLMEmbedder()


def embed_text(text: str) -> list[float]:
    return get_embedder().embed(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed_many(texts)
