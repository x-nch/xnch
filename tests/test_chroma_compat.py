"""Tests for the agentmemory/chromadb compatibility shim."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from xnch.memory._chroma_compat import _apply


class FakeCollection:
    """Minimal stand-in for chromadb Collection with keyword-only safety."""

    def __init__(self) -> None:
        self.last_kwargs = {}
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        self.last_kwargs = kwargs
        return {"ids": []}

    def add(self, **kwargs):
        self.calls.append(("add", kwargs))
        self.last_kwargs = kwargs

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))
        self.last_kwargs = kwargs


class FakeChromaCollectionMemory:
    """Mirrors agentmemory's ChromaCollectionMemory shape."""

    def __init__(self, collection) -> None:
        self.collection = collection


def _patched_memory():
    """Apply the shim to a fake class and return (memory, collection)."""
    coll = FakeCollection()
    mem = FakeChromaCollectionMemory(coll)
    return mem, coll


def test_query_passes_n_results_as_keyword():
    """n_results must reach the collection as a keyword arg."""
    mem, coll = _patched_memory()
    ChromaCollectionMemory = FakeChromaCollectionMemory

    # Re-run the shim targeting the fake class
    with patch("agentmemory.chroma_client.ChromaCollectionMemory", ChromaCollectionMemory):
        _apply()

    query = ChromaCollectionMemory.query
    query(mem, query_texts=["test"], n_results=5, where=None, where_document=None)
    assert coll.calls[0][0] == "query"
    kwargs = coll.calls[0][1]
    assert kwargs["n_results"] == 5
    assert kwargs["query_texts"] == ["test"]


def test_query_positional_args_still_supported():
    """The shim keeps the agentmemory positional signature."""
    mem, coll = _patched_memory()
    ChromaCollectionMemory = FakeChromaCollectionMemory

    with patch("agentmemory.chroma_client.ChromaCollectionMemory", ChromaCollectionMemory):
        _apply()

    query = ChromaCollectionMemory.query
    query(mem, None, ["test"], 7, None, None, ["metadatas"])
    kwargs = coll.calls[0][1]
    assert kwargs["n_results"] == 7
    assert kwargs["query_texts"] == ["test"]


def test_add_maps_documents_correctly():
    """documents must not land in the embeddings slot."""
    mem, coll = _patched_memory()
    ChromaCollectionMemory = FakeChromaCollectionMemory

    with patch("agentmemory.chroma_client.ChromaCollectionMemory", ChromaCollectionMemory):
        _apply()

    add = ChromaCollectionMemory.add
    add(mem, ["id-1"], documents=["doc"], metadatas=[{"k": "v"}], embeddings=None)
    kwargs = coll.calls[0][1]
    assert kwargs["documents"] == ["doc"]
    assert kwargs["metadatas"] == [{"k": "v"}]
    assert kwargs["ids"] == ["id-1"]


def test_update_maps_documents_correctly():
    mem, coll = _patched_memory()
    ChromaCollectionMemory = FakeChromaCollectionMemory

    with patch("agentmemory.chroma_client.ChromaCollectionMemory", ChromaCollectionMemory):
        _apply()

    update = ChromaCollectionMemory.update
    update(mem, ["id-1"], documents=["doc"], metadatas=[{"k": "v"}], embeddings=None)
    kwargs = coll.calls[0][1]
    assert kwargs["documents"] == ["doc"]
    assert kwargs["embeddings"] is None
