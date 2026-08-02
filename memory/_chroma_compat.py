"""Compatibility shim for agentmemory 0.4.8 + chromadb >= 1.0.

chromadb 1.0 reordered Collection.query/add/update positional parameters
(inserting query_images/query_uris/ids before n_results, and embeddings
before documents). agentmemory 0.4.8 forwards positional args matching the
pre-1.0 API, which silently corrupts calls — e.g. n_results lands in the
query_images slot, raising "Expected image to be a numpy array, got 1".

This module monkeypatches agentmemory's ChromaCollectionMemory wrappers to
pass keyword arguments, keeping them correct across chromadb versions.
"""

from __future__ import annotations


def _apply() -> None:
    try:
        from agentmemory.chroma_client import ChromaCollectionMemory
    except ImportError:
        return

    def query(
        self,
        query_embeddings=None,
        query_texts=None,
        n_results: int = 10,
        where=None,
        where_document=None,
        include=("metadatas", "documents", "distances"),
    ):
        return self.collection.query(
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
        )

    def add(self, ids, documents=None, metadatas=None, embeddings=None):
        return self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def update(self, ids, documents=None, metadatas=None, embeddings=None):
        return self.collection.update(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    ChromaCollectionMemory.query = query
    ChromaCollectionMemory.add = add
    ChromaCollectionMemory.update = update


_apply()
