"""
Index building: ingestion -> chunking -> vector store sync.

Kept separate from the Streamlit UI so the full pipeline can be tested
against a real (temporary) Chroma collection.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from app.chunking import DocumentChunker
from app.config import Settings, settings as default_settings
from app.demo import require_write_access
from app.ingestion import PDFIngestor
from app.utils import get_logger
from app.vector_store import IndexingError, SourceStats, VectorStoreManager

logger = get_logger(__name__)

# Document states shown in the Knowledge Base.
INDEXED = "indexed"
CHANGED = "changed"  # file on disk differs from the indexed version
NOT_INDEXED = "not_indexed"
DUPLICATE = "duplicate"  # identical content is indexed under another name
NO_TEXT = "no_text"  # last indexing run found no extractable text
UNREADABLE = "unreadable"
REMOVED = "removed"  # indexed, but the file is no longer on disk


@dataclass
class DocumentStatus:
    name: str
    status: str
    chunks: int = 0
    pages: int = 0
    size_bytes: Optional[int] = None
    duplicate_of: Optional[str] = None

    @property
    def needs_indexing(self) -> bool:
        return self.status in (CHANGED, NOT_INDEXED, REMOVED)


def document_statuses(
    pdf_paths: Iterable[Path],
    file_hash: Callable[[Path], Optional[str]],
    indexed: dict[str, SourceStats],
    no_text_files: Iterable[str] = (),
) -> list[DocumentStatus]:
    """Compare the PDFs on disk with what the index contains.

    `file_hash(path)` returns the file's content hash, or None if it can't
    be read. `no_text_files` are names the last indexing run reported as
    having no extractable text (they legitimately have no chunks).
    """
    no_text = set(no_text_files)
    paths = list(pdf_paths)
    on_disk = {p.name for p in paths}
    # Only count a file as a duplicate of an indexed copy that still exists;
    # a renamed file must not look like a duplicate of its own old name.
    indexed_by_hash = {
        stats.file_hash: name for name, stats in indexed.items() if stats.file_hash and name in on_disk
    }
    statuses: list[DocumentStatus] = []

    for path in paths:
        name = path.name
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        digest = file_hash(path)
        stats = indexed.get(name)
        if digest is None:
            status = DocumentStatus(name, UNREADABLE, size_bytes=size)
        elif stats is not None and stats.file_hash == digest:
            status = DocumentStatus(name, INDEXED, stats.chunks, stats.pages, size)
        elif stats is not None:
            status = DocumentStatus(name, CHANGED, stats.chunks, stats.pages, size)
        elif digest in indexed_by_hash:
            status = DocumentStatus(name, DUPLICATE, size_bytes=size, duplicate_of=indexed_by_hash[digest])
        elif name in no_text:
            status = DocumentStatus(name, NO_TEXT, size_bytes=size)
        else:
            status = DocumentStatus(name, NOT_INDEXED, size_bytes=size)
        statuses.append(status)

    for name in sorted(set(indexed) - on_disk):
        statuses.append(DocumentStatus(name, REMOVED, indexed[name].chunks, indexed[name].pages))
    return statuses


def build_index(
    vsm: VectorStoreManager, settings: Optional[Settings] = None, rebuild: bool = False
) -> dict:
    """Ingest every PDF in the data directory and bring the index in line
    with it. Returns a stats dict for display.

    Incremental mode (default) embeds only new or changed chunks and
    removes chunks from edited, renamed, or deleted PDFs. `rebuild=True`
    re-embeds everything into a staging collection before swapping it in.

    Raises IndexingError with a user-facing message on failure; the
    existing index is never silently destroyed. Raises DemoModeRestriction
    when PUBLIC_DEMO_MODE is enabled.
    """
    s = settings or default_settings
    # Server-side guard (the UI also hides these actions in demo mode).
    require_write_access(s, "Rebuilding the index" if rebuild else "Updating the index")
    try:
        pages, report = PDFIngestor(s).ingest_directory()
        documents = DocumentChunker(s).chunk_pages(pages)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion failed.")
        raise IndexingError(f"Reading the PDFs failed: {exc}. The existing index was left unchanged.") from exc

    removed = unchanged = 0
    try:
        if rebuild:
            added = vsm.rebuild(documents)
        else:
            # Files that exist but could not be read this run keep their
            # previously indexed chunks rather than being dropped.
            result = vsm.sync_documents(documents, report.file_hashes, keep_sources=report.failed_files)
            added, removed, unchanged = result.added, result.removed, result.unchanged
    except IndexingError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Indexing failed.")
        raise IndexingError(f"Indexing failed: {exc}. The existing index was left unchanged.") from exc

    return {
        "processed_files": report.processed_files,
        "skipped_duplicates": report.skipped_duplicates,
        "failed_files": report.failed_files,
        "empty_or_scanned_files": report.empty_or_scanned_files,
        "total_pages": report.total_pages,
        "total_chunks_this_run": len(documents),
        "chunks_added": added,
        "chunks_removed": removed,
        "chunks_unchanged": unchanged,
    }


# Demo seeding outcomes.
SEED_DISABLED = "disabled"  # PUBLIC_DEMO_MODE is off; nothing is touched
SEED_ALREADY_INDEXED = "already_indexed"  # the index has content; nothing is touched
SEED_SEEDED = "seeded"

# Serializes seeding within a process so concurrent sessions can't index the
# demo corpus at the same time (chunk IDs are deterministic, so a repeat
# would be harmless anyway, just wasted work).
_seed_lock = threading.Lock()


@dataclass
class DemoSeedResult:
    status: str
    chunks_added: int = 0
    documents: tuple[str, ...] = ()


def seed_demo_index(vsm: VectorStoreManager, settings: Optional[Settings] = None) -> DemoSeedResult:
    """In public demo mode, index the bundled demo PDFs if the index is empty.

    Add-only by design: it never deletes, rebuilds, or modifies an index
    that already has content, and does nothing outside demo mode, so a
    developer's local corpus is never touched. Safe to call on every app
    start; Streamlit Community Cloud may reset the runtime, in which case
    the bundled PDFs are re-indexed from scratch.

    Raises IndexingError with a user-facing message if seeding fails.
    """
    s = settings or default_settings
    if not s.public_demo_mode:
        return DemoSeedResult(SEED_DISABLED)

    with _seed_lock:
        if vsm.count() > 0:
            return DemoSeedResult(SEED_ALREADY_INDEXED)
        try:
            pages, report = PDFIngestor(s).ingest_directory()
            documents = DocumentChunker(s).chunk_pages(pages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reading the demo corpus failed.")
            raise IndexingError(f"Reading the demo documents in {s.data_dir} failed: {exc}") from exc
        if not documents:
            raise IndexingError(f"No demo documents with extractable text were found in {s.data_dir}.")
        try:
            added = vsm.add_documents(documents, skip_duplicates=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Indexing the demo corpus failed.")
            raise IndexingError(f"Indexing the demo documents failed: {exc}") from exc

    logger.info("Seeded the demo index with %d chunks from %d documents.", added, len(report.processed_files))
    return DemoSeedResult(SEED_SEEDED, added, tuple(report.processed_files))
