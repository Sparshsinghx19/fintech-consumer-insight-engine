"""
Document ingestion for the FinTech Consumer Insight Engine.

Responsible for discovering PDFs in the configured raw-data directory,
extracting text + metadata per page, and gracefully handling:
  - empty PDFs
  - corrupted / unreadable PDFs
  - scanned PDFs with no extractable text layer
  - duplicate documents (via file content hash)
  - documents with unusual formatting
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import Settings, settings as default_settings
from app.utils import clean_text, get_logger, sha256_file

logger = get_logger(__name__)


@dataclass
class PageRecord:
    """A single extracted page, ready for chunking."""

    text: str
    source: str  # file name
    source_path: str  # absolute/relative path on disk
    page: int  # 1-indexed page number
    document_type: str = "pdf"
    file_hash: str = ""


@dataclass
class IngestionReport:
    """Summary statistics produced after an ingestion run."""

    processed_files: List[str] = field(default_factory=list)
    skipped_duplicates: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    empty_or_scanned_files: List[str] = field(default_factory=list)
    total_pages: int = 0
    total_pages_with_text: int = 0

    @property
    def total_documents(self) -> int:
        return len(self.processed_files)


class PDFIngestor:
    """Discovers and extracts text/metadata from PDFs on disk."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or default_settings

    def discover_pdfs(self, directory: Optional[Path] = None) -> List[Path]:
        directory = directory or self.settings.data_dir
        if not directory.exists():
            logger.warning("Data directory %s does not exist yet.", directory)
            return []
        return sorted(p for p in directory.glob("*.pdf") if p.is_file())

    def extract_pages(self, pdf_path: Path) -> List[PageRecord]:
        """Extract text per page from a single PDF.

        Returns an empty list (rather than raising) for corrupted or
        fully-scanned documents so the caller can continue processing the
        rest of the corpus. Errors are logged with the offending file name.
        """
        pages: List[PageRecord] = []
        try:
            file_hash = sha256_file(str(pdf_path))
        except OSError as exc:
            logger.error("Could not read file %s: %s", pdf_path, exc)
            return []

        try:
            reader = PdfReader(str(pdf_path))
        except (PdfReadError, OSError, ValueError) as exc:
            logger.error("Corrupted or unreadable PDF %s: %s", pdf_path.name, exc)
            return []

        if reader.is_encrypted:
            try:
                # Attempt a blank-password decrypt (common for "restricted"
                # but not truly password-protected exports).
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001 - decrypt raises various types
                logger.error("Encrypted PDF could not be opened %s: %s", pdf_path.name, exc)
                return []

        num_pages = len(reader.pages)
        if num_pages == 0:
            logger.warning("Empty PDF (0 pages): %s", pdf_path.name)
            return []

        for idx in range(num_pages):
            try:
                raw_text = reader.pages[idx].extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - pypdf can raise many types
                logger.warning("Failed to extract page %d of %s: %s", idx + 1, pdf_path.name, exc)
                raw_text = ""

            cleaned = clean_text(raw_text)
            pages.append(
                PageRecord(
                    text=cleaned,
                    source=pdf_path.name,
                    source_path=str(pdf_path),
                    page=idx + 1,
                    document_type="pdf",
                    file_hash=file_hash,
                )
            )
        return pages

    def ingest_directory(self, directory: Optional[Path] = None) -> tuple[List[PageRecord], IngestionReport]:
        """Ingest every PDF in the directory, skipping duplicate files by
        content hash (not just filename)."""
        directory = directory or self.settings.data_dir
        report = IngestionReport()
        all_pages: List[PageRecord] = []
        seen_hashes: dict[str, str] = {}

        for pdf_path in self.discover_pdfs(directory):
            try:
                file_hash = sha256_file(str(pdf_path))
            except OSError as exc:
                logger.error("Skipping unreadable file %s: %s", pdf_path, exc)
                report.failed_files.append(pdf_path.name)
                continue

            if file_hash in seen_hashes:
                logger.info(
                    "Skipping duplicate document %s (identical content to %s)",
                    pdf_path.name,
                    seen_hashes[file_hash],
                )
                report.skipped_duplicates.append(pdf_path.name)
                continue

            pages = self.extract_pages(pdf_path)
            if not pages:
                report.failed_files.append(pdf_path.name)
                continue

            seen_hashes[file_hash] = pdf_path.name
            report.processed_files.append(pdf_path.name)
            report.total_pages += len(pages)

            pages_with_text = [p for p in pages if p.text.strip()]
            report.total_pages_with_text += len(pages_with_text)

            if not pages_with_text:
                logger.warning(
                    "%s produced no extractable text on any page (likely a scanned "
                    "document without OCR). It will be indexed with empty content.",
                    pdf_path.name,
                )
                report.empty_or_scanned_files.append(pdf_path.name)

            all_pages.extend(pages)

        return all_pages, report


def filter_pages_with_content(pages: Iterable[PageRecord]) -> List[PageRecord]:
    """Drop pages with no usable text (e.g. blank or scanned pages) before
    chunking, since empty chunks add no retrieval value."""
    return [p for p in pages if p.text and p.text.strip()]
