"""Tests for chunking: size/overlap config, metadata preservation."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.chunking import DocumentChunker
from app.config import Settings
from app.ingestion import PageRecord


@pytest.fixture
def small_chunk_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vs",
        chunk_size=50,
        chunk_overlap=10,
    )


def test_chunk_pages_preserves_metadata(small_chunk_settings: Settings):
    pages = [
        PageRecord(
            text="This is sentence one. " * 10,
            source="report.pdf",
            source_path="/data/raw/report.pdf",
            page=7,
            document_type="pdf",
            file_hash="abc123",
        )
    ]
    chunker = DocumentChunker(small_chunk_settings)
    docs = chunker.chunk_pages(pages)

    assert len(docs) > 1  # long text should split into multiple chunks
    for doc in docs:
        assert doc.metadata["source"] == "report.pdf"
        assert doc.metadata["page"] == 7
        assert doc.metadata["document_type"] == "pdf"
        assert doc.metadata["file_hash"] == "abc123"
        assert "chunk_hash" in doc.metadata
        assert "chunk_index" in doc.metadata


def test_chunk_pages_respects_chunk_size(small_chunk_settings: Settings):
    pages = [
        PageRecord(text="word " * 200, source="a.pdf", source_path="a.pdf", page=1),
    ]
    chunker = DocumentChunker(small_chunk_settings)
    docs = chunker.chunk_pages(pages)
    # Allow a little slack since the splitter breaks on separators, not
    # mid-word, but no chunk should be wildly larger than configured.
    for doc in docs:
        assert len(doc.page_content) <= small_chunk_settings.chunk_size + 20


def test_chunk_pages_skips_empty_pages(small_chunk_settings: Settings):
    pages = [
        PageRecord(text="", source="a.pdf", source_path="a.pdf", page=1),
        PageRecord(text="   ", source="a.pdf", source_path="a.pdf", page=2),
    ]
    chunker = DocumentChunker(small_chunk_settings)
    docs = chunker.chunk_pages(pages)
    assert docs == []


def test_chunk_hash_differs_per_chunk(small_chunk_settings: Settings):
    pages = [
        PageRecord(text="Alpha content sentence. " * 8, source="a.pdf", source_path="a.pdf", page=1),
    ]
    chunker = DocumentChunker(small_chunk_settings)
    docs = chunker.chunk_pages(pages)
    hashes = [d.metadata["chunk_hash"] for d in docs]
    assert len(hashes) == len(set(hashes))  # all unique
