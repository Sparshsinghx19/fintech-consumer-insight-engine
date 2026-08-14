"""Tests for PDF ingestion: extraction, metadata, duplicates, error handling."""
from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.config import Settings
from app.ingestion import PDFIngestor, filter_pages_with_content


def _make_pdf(path: Path, num_pages: int = 1, with_text: bool = True) -> None:
    """Create a minimal real PDF file for tests.

    pypdf's PdfWriter can create blank pages; it cannot easily draw text
    without extra dependencies, so "with_text" pages are still blank at
    the pypdf level - text-extraction behavior is exercised in the
    extraction-empty-page path, and text-bearing behavior is validated
    via the clean_text/metadata pipeline in chunking tests using
    synthetic PageRecords.
    """
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as f:
        writer.write(f)


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    return Settings(data_dir=data_dir, processed_dir=tmp_path / "processed", vectorstore_dir=tmp_path / "vs")


def test_discover_pdfs_finds_files(tmp_settings: Settings):
    _make_pdf(tmp_settings.data_dir / "a.pdf")
    _make_pdf(tmp_settings.data_dir / "b.pdf")
    (tmp_settings.data_dir / "not_a_pdf.txt").write_text("hello")

    ingestor = PDFIngestor(tmp_settings)
    found = ingestor.discover_pdfs()
    names = sorted(p.name for p in found)
    assert names == ["a.pdf", "b.pdf"]


def test_extract_pages_returns_page_records_with_metadata(tmp_settings: Settings):
    pdf_path = tmp_settings.data_dir / "sample.pdf"
    _make_pdf(pdf_path, num_pages=3)

    ingestor = PDFIngestor(tmp_settings)
    pages = ingestor.extract_pages(pdf_path)

    assert len(pages) == 3
    for i, page in enumerate(pages, start=1):
        assert page.source == "sample.pdf"
        assert page.page == i
        assert page.document_type == "pdf"
        assert page.file_hash  # non-empty hash


def test_extract_pages_handles_corrupted_file(tmp_settings: Settings):
    bad_pdf = tmp_settings.data_dir / "corrupted.pdf"
    bad_pdf.write_bytes(b"this is not a valid pdf file")

    ingestor = PDFIngestor(tmp_settings)
    pages = ingestor.extract_pages(bad_pdf)
    assert pages == []


def test_extract_pages_handles_empty_pdf(tmp_settings: Settings):
    empty_pdf = tmp_settings.data_dir / "empty.pdf"
    writer = PdfWriter()  # zero pages
    with open(empty_pdf, "wb") as f:
        writer.write(f)

    ingestor = PDFIngestor(tmp_settings)
    pages = ingestor.extract_pages(empty_pdf)
    assert pages == []


def test_ingest_directory_skips_duplicate_content(tmp_settings: Settings):
    pdf1 = tmp_settings.data_dir / "doc1.pdf"
    _make_pdf(pdf1, num_pages=1)
    # Byte-identical duplicate under a different name.
    pdf2 = tmp_settings.data_dir / "doc1_copy.pdf"
    pdf2.write_bytes(pdf1.read_bytes())

    ingestor = PDFIngestor(tmp_settings)
    pages, report = ingestor.ingest_directory()

    assert report.total_documents == 1
    assert "doc1_copy.pdf" in report.skipped_duplicates


def test_ingest_directory_reports_failed_files(tmp_settings: Settings):
    (tmp_settings.data_dir / "broken.pdf").write_bytes(b"not a real pdf")
    ingestor = PDFIngestor(tmp_settings)
    pages, report = ingestor.ingest_directory()
    assert "broken.pdf" in report.failed_files
    assert report.total_documents == 0


def test_filter_pages_with_content_drops_blank_pages():
    from app.ingestion import PageRecord

    records = [
        PageRecord(text="Real content here.", source="a.pdf", source_path="a.pdf", page=1),
        PageRecord(text="   ", source="a.pdf", source_path="a.pdf", page=2),
        PageRecord(text="", source="a.pdf", source_path="a.pdf", page=3),
    ]
    filtered = filter_pages_with_content(records)
    assert len(filtered) == 1
    assert filtered[0].page == 1
