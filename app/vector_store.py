"""
Persistent ChromaDB vector store management.

Supports:
  - initial indexing (build from scratch)
  - incremental sync (embed new/changed chunks, remove chunks of edited,
    renamed, or deleted PDFs, never re-embed unchanged ones)
  - full re-indexing / rebuild via a staging collection
  - duplicate-chunk detection via content hash stored in metadata
  - batched writes within Chroma's maximum batch size
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import Settings, settings as default_settings
from app.embeddings import get_embedding_function
from app.utils import get_logger

logger = get_logger(__name__)


_STAGING_SUFFIX = "__rebuild"
# Chroma rejects writes larger than its max batch size (5461 on the
# default SQLite backend); stay safely under it.
DEFAULT_MAX_BATCH_SIZE = 5000


class IndexingError(RuntimeError):
    """Indexing could not complete. The message is safe to show to users
    and states what happened to the existing index."""


@dataclass
class SyncResult:
    added: int
    removed: int
    unchanged: int


def _batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _max_batch_size(store: Chroma) -> int:
    try:
        size = store._client.get_max_batch_size()
    except Exception:  # noqa: BLE001
        return DEFAULT_MAX_BATCH_SIZE
    if isinstance(size, int) and size > 0:
        return min(size, DEFAULT_MAX_BATCH_SIZE)
    return DEFAULT_MAX_BATCH_SIZE


def _unique_by_id(documents: List[Document]) -> List[Document]:
    """Drop repeated chunk IDs; Chroma rejects duplicate IDs in one write."""
    seen: set[str] = set()
    unique = []
    for doc in documents:
        chunk_id = doc.metadata["chunk_hash"]
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(doc)
    return unique


def _chroma_client_settings() -> ChromaSettings:
    """Chroma client settings with anonymized telemetry disabled, so the
    vector store makes no outbound network calls. `is_persistent` must be
    set explicitly: when client settings are supplied, langchain_chroma
    does not enable on-disk persistence by itself."""
    return ChromaSettings(anonymized_telemetry=False, is_persistent=True)


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
                client_settings=_chroma_client_settings(),
            )
        return self._store

    def rebuild(self, documents: List[Document]) -> int:
        """Re-embed everything from scratch. Use when chunking/embedding
        configuration has changed.

        The replacement is built in a separate staging collection first;
        the live collection is only swapped out once every chunk has been
        embedded successfully. On failure the existing index is untouched.
        Refuses to replace the index with nothing.
        """
        docs = _unique_by_id(documents)
        if not docs:
            raise IndexingError(
                "Rebuild produced no indexable chunks (no PDFs with extractable text were found). "
                "The existing index was left unchanged."
            )

        name = self.settings.collection_name
        staging_name = f"{name}{_STAGING_SUFFIX}"
        client = self.get_store()._client
        self._drop_collection_if_exists(client, staging_name)  # leftover from an interrupted rebuild
        staging = Chroma(collection_name=staging_name, embedding_function=self._embeddings, client=client)

        try:
            self._write(staging, docs)
        except Exception as exc:  # noqa: BLE001
            self._drop_collection_if_exists(client, staging_name)
            raise IndexingError(
                f"Rebuild failed while embedding chunks: {exc}. The existing index was left unchanged."
            ) from exc

        try:
            self._drop_collection_if_exists(client, name)
            staging._collection.modify(name=name)
        except Exception as exc:  # noqa: BLE001
            raise IndexingError(
                f"Rebuilt index could not be swapped into place: {exc}. "
                f"The new chunks are in the '{staging_name}' collection; run the rebuild again."
            ) from exc
        finally:
            self._store = None

        logger.info("Rebuilt the vector store with %d chunks.", len(docs))
        return len(docs)

    @staticmethod
    def _drop_collection_if_exists(client, name: str) -> None:
        if name in {c.name for c in client.list_collections()}:
            client.delete_collection(name)

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

        to_add = _unique_by_id(documents)
        if skip_duplicates:
            existing = self.existing_chunk_hashes()
            to_add = [d for d in to_add if d.metadata.get("chunk_hash") not in existing]
            skipped = len(documents) - len(to_add)
            if skipped:
                logger.info("Skipping %d already-indexed chunks (incremental indexing).", skipped)

        if not to_add:
            logger.info("No new chunks to embed; vector store already up to date.")
            return 0

        self._write(self.get_store(), to_add)
        logger.info("Added %d new chunks to the vector store.", len(to_add))
        return len(to_add)

    def sync_documents(
        self,
        documents: List[Document],
        current_files: dict[str, str],
        keep_sources: Iterable[str] = (),
        allow_empty: bool = False,
    ) -> SyncResult:
        """Make the index match the current set of PDFs.

        `documents` are the chunks of the current files, and `current_files`
        maps each successfully ingested file name to its content hash.
        Chunk IDs are deterministic (source, page, chunk index, text), so
        the current chunks define exactly which IDs should exist:

          - new or changed chunks are embedded and added;
          - unchanged chunks are left alone (not re-embedded);
          - every other chunk - from edited, renamed, or deleted PDFs - is
            removed, so stale text can no longer be retrieved or cited.

        Chunks whose source is in `keep_sources` (files present on disk but
        unreadable this run) are kept untouched. New chunks are written
        before obsolete ones are deleted, so a failed embedding never loses
        existing data. Refuses to empty a non-empty index when no source
        files were found, unless `allow_empty` is set.
        """
        store = self.get_store()
        try:
            existing = store.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise IndexingError(f"Could not read the existing index: {exc}") from exc
        existing_ids = existing.get("ids") or []
        existing_metas = existing.get("metadatas") or [{}] * len(existing_ids)

        keep = set(keep_sources)
        if existing_ids and not current_files and not keep and not allow_empty:
            raise IndexingError(
                f"No PDFs were found in {self.settings.data_dir}, so indexing would remove all "
                f"{len(existing_ids)} indexed chunks. The existing index was left unchanged. "
                "Delete documents individually to remove them from the index."
            )

        desired = {d.metadata["chunk_hash"]: d for d in _unique_by_id(documents)}
        up_to_date: set[str] = set()
        stale: list[str] = []
        for chunk_id, meta in zip(existing_ids, existing_metas):
            meta = meta or {}
            doc = desired.get(chunk_id)
            if doc is not None and meta.get("file_hash") == doc.metadata.get("file_hash"):
                up_to_date.add(chunk_id)
            elif doc is None and meta.get("source") not in keep:
                stale.append(chunk_id)
            # A desired chunk with outdated metadata is re-written below.

        to_add = [d for chunk_id, d in desired.items() if chunk_id not in up_to_date]
        try:
            self._write(store, to_add)
        except Exception as exc:  # noqa: BLE001
            raise IndexingError(
                f"Embedding new chunks failed: {exc}. Previously indexed chunks were left in place."
            ) from exc

        try:
            for batch in _batches(stale, _max_batch_size(store)):
                store.delete(ids=batch)
        except Exception as exc:  # noqa: BLE001
            raise IndexingError(
                f"New chunks were added, but obsolete chunks could not be removed: {exc}. "
                "Run indexing again to finish cleaning up."
            ) from exc

        result = SyncResult(added=len(to_add), removed=len(stale), unchanged=len(up_to_date))
        logger.info(
            "Index sync: %d added, %d removed, %d unchanged.", result.added, result.removed, result.unchanged
        )
        return result

    def _write(self, store: Chroma, documents: List[Document]) -> None:
        """Upsert documents in batches no larger than Chroma accepts."""
        for batch in _batches(documents, _max_batch_size(store)):
            store.add_documents(documents=batch, ids=[d.metadata["chunk_hash"] for d in batch])

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
            for batch in _batches(ids, _max_batch_size(store)):
                store.delete(ids=batch)
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
