"""
Index building: ingestion -> chunking -> vector store sync.

Kept separate from the Streamlit UI so the full pipeline can be tested
against a real (temporary) Chroma collection.
"""
from __future__ import annotations

from typing import Optional

from app.chunking import DocumentChunker
from app.config import Settings, settings as default_settings
from app.ingestion import PDFIngestor
from app.utils import get_logger
from app.vector_store import IndexingError, VectorStoreManager

logger = get_logger(__name__)


def build_index(
    vsm: VectorStoreManager, settings: Optional[Settings] = None, rebuild: bool = False
) -> dict:
    """Ingest every PDF in the data directory and bring the index in line
    with it. Returns a stats dict for display.

    Incremental mode (default) embeds only new or changed chunks and
    removes chunks from edited, renamed, or deleted PDFs. `rebuild=True`
    re-embeds everything into a staging collection before swapping it in.

    Raises IndexingError with a user-facing message on failure; the
    existing index is never silently destroyed.
    """
    s = settings or default_settings
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
