"""
Persistent ChromaDB vector store management.

Supports:
  - initial indexing (build from scratch)
  - incremental indexing (add new documents without re-embedding existing ones)
  - full re-indexing / rebuild
  - duplicate-chunk detection via content hash stored in metadata
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import Settings, settings as default_settings
from app.embeddings import get_embedding_function
from app.utils import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or default_settings
        self.settings.ensure_directories()
        self._embeddings = get_embedding_function(self.settings)
        self._store: Optional[Chroma] = None

    # -- lifecycle -----------------------------------------------------
    def get_store(self) -> Chroma:
        """Lazily open (or create) the persistent Chroma collection."""
        if self._store is None:
            self._store = Chroma(
                collection_name=self.settings.collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(self.settings.vectorstore_dir),
            )
        return self._store

    def rebuild(self, documents: List[Document]) -> int:
        """Drop the existing collection and re-embed everything from
        scratch. Use when chunking/embedding configuration has changed."""
        self.reset()
        return self.add_documents(documents, skip_duplicates=False)

    def reset(self) -> None:
        """Delete the persisted collection entirely."""
        store = self.get_store()
        try:
            store.delete_collection()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not delete existing collection cleanly: %s", exc)
        self._store = None

    # -- indexing --------------------------------------------------------
    def existing_chunk_hashes(self) -> set[str]:
        """Fetch the set of chunk_hash values already stored, so incremental
        indexing can skip documents that were already embedded."""
        store = self.get_store()
        try:
            result = store.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read existing metadata for dedup check: %s", exc)
            return set()
        hashes = set()
        for meta in result.get("metadatas", []) or []:
            if meta and meta.get("chunk_hash"):
                hashes.add(meta["chunk_hash"])
        return hashes

    def add_documents(self, documents: List[Document], skip_duplicates: bool = True) -> int:
        """Add chunks to the vector store. When `skip_duplicates` is True
        (the default, used for incremental indexing), chunks whose
        `chunk_hash` already exists in the collection are not re-embedded."""
        if not documents:
            logger.info("No documents supplied to add_documents; nothing to do.")
            return 0

        to_add = documents
        if skip_duplicates:
            existing = self.existing_chunk_hashes()
            to_add = [d for d in documents if d.metadata.get("chunk_hash") not in existing]
            skipped = len(documents) - len(to_add)
            if skipped:
                logger.info("Skipping %d already-indexed chunks (incremental indexing).", skipped)

        if not to_add:
            logger.info("No new chunks to embed; vector store already up to date.")
            return 0

        store = self.get_store()
        ids = [d.metadata["chunk_hash"] for d in to_add]
        store.add_documents(documents=to_add, ids=ids)
        logger.info("Added %d new chunks to the vector store.", len(to_add))
        return len(to_add)

    # -- stats -----------------------------------------------------------
    def count(self) -> int:
        try:
            return self.get_store()._collection.count()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read collection count: %s", exc)
            return 0

    def delete_by_source(self, source: str) -> int:
        """Remove every chunk belonging to a given source document (by
        file name) from the vector store. Returns the number of chunks
        removed. Use this when a PDF is deleted from data/raw/ so stale
        chunks don't linger in the index or get returned by search."""
        store = self.get_store()
        try:
            result = store.get(where={"source": source}, include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not look up chunks for source '%s': %s", source, exc)
            return 0

        ids = result.get("ids", []) or []
        if not ids:
            logger.info("No indexed chunks found for source '%s'.", source)
            return 0

        try:
            store.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to delete chunks for source '%s': %s", source, exc)
            return 0

        logger.info("Deleted %d chunks for source '%s' from the vector store.", len(ids), source)
        return len(ids)

    def indexed_sources(self) -> set[str]:
        store = self.get_store()
        try:
            result = store.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read indexed sources: %s", exc)
            return set()
        sources = set()
        for meta in result.get("metadatas", []) or []:
            if meta and meta.get("source"):
                sources.add(meta["source"])
        return sources

    def is_persisted(self) -> bool:
        """Whether a Chroma collection already exists on disk."""
        return any(Path(self.settings.vectorstore_dir).glob("*")) and self.count() > 0
