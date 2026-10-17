"""Regression tests for index consistency and indexing safety, run end to
end (real text-bearing PDFs -> ingestion -> chunking -> an isolated real
Chroma collection in a temporary directory)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.indexing import build_index
from app.vector_store import IndexingError, VectorStoreManager
from tests.helpers import HashingEmbeddings, make_text_pdf

TRUST_PAGES = [
    "Sixty two percent of consumers cite privacy concerns with digital wallets.",
    "Hidden fees are a major barrier to mobile payment adoption.",
]
FEES_PAGES = ["Transparent transaction fees increase willingness to pay by card."]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path / "raw", processed_dir=tmp_path / "processed", vectorstore_dir=tmp_path / "vs")
    s.ensure_directories()
    return s


@pytest.fixture
def embeddings() -> HashingEmbeddings:
    return HashingEmbeddings()


@pytest.fixture
def vsm(settings: Settings, embeddings: HashingEmbeddings) -> VectorStoreManager:
    with patch("app.vector_store.get_embedding_function", return_value=embeddings):
        return VectorStoreManager(settings)


def _indexed(vsm: VectorStoreManager) -> list[tuple[str, dict]]:
    result = vsm.get_store().get(include=["documents", "metadatas"])
    return list(zip(result["documents"], result["metadatas"]))


def _texts_for(vsm: VectorStoreManager, source: str) -> list[str]:
    return [text for text, meta in _indexed(vsm) if meta["source"] == source]


def test_initial_index_preserves_metadata(settings, vsm):
    make_text_pdf(settings.data_dir / "trust.pdf", TRUST_PAGES)

    stats = build_index(vsm, settings)

    assert stats["chunks_added"] == 2
    entries = _indexed(vsm)
    assert sorted(meta["page"] for _, meta in entries) == [1, 2]
    for text, meta in entries:
        assert meta["source"] == "trust.pdf"
        assert meta["chunk_index"] == 0
        assert meta["chunk_hash"] and meta["file_hash"]
    assert any("privacy concerns" in text for text, _ in entries)


def test_unchanged_pdfs_are_not_reindexed_or_duplicated(settings, vsm):
    make_text_pdf(settings.data_dir / "trust.pdf", TRUST_PAGES)
    build_index(vsm, settings)

    stats = build_index(vsm, settings)
    stats_again = build_index(vsm, settings)

    assert (stats["chunks_added"], stats["chunks_removed"], stats["chunks_unchanged"]) == (0, 0, 2)
    assert stats_again["chunks_added"] == 0
    assert vsm.count() == 2


def test_modified_pdf_replaces_obsolete_chunks(settings, vsm):
    pdf = make_text_pdf(settings.data_dir / "trust.pdf", TRUST_PAGES)
    build_index(vsm, settings)

    make_text_pdf(pdf, ["Forty percent of consumers cite privacy concerns.", TRUST_PAGES[1]])
    stats = build_index(vsm, settings)

    texts = _texts_for(vsm, "trust.pdf")
    assert vsm.count() == 2
    assert not any("Sixty two percent" in t for t in texts)
    assert any("Forty percent" in t for t in texts)
    assert (stats["chunks_added"], stats["chunks_removed"]) == (2, 1)  # page 2 text unchanged but its file_hash is refreshed
    new_hash = {meta["file_hash"] for _, meta in _indexed(vsm)}
    assert len(new_hash) == 1  # every chunk carries the current file's hash


def test_renamed_pdf_leaves_no_chunks_under_old_name(settings, vsm):
    pdf = make_text_pdf(settings.data_dir / "trust.pdf", TRUST_PAGES)
    build_index(vsm, settings)

    pdf.rename(settings.data_dir / "trust_2024.pdf")
    build_index(vsm, settings)

    assert vsm.indexed_sources() == {"trust_2024.pdf"}
    assert vsm.count() == 2


def test_deleted_pdf_chunks_are_removed(settings, vsm):
    trust = make_text_pdf(settings.data_dir / "trust.pdf", TRUST_PAGES)
    make_text_pdf(settings.data_dir / "fees.pdf", FEES_PAGES)
    build_index(vsm, settings)

    trust.unlink()
    stats = build_index(vsm, settings)

    assert vsm.indexed_sources() == {"fees.pdf"}
    assert stats["chunks_removed"] == 2
    results = vsm.get_store().similarity_search("privacy concerns digital wallets", k=5)
    assert all(doc.metadata["source"] == "fees.pdf" for doc in results)


def test_identical_copy_under_new_name_is_not_indexed_twice(settings, vsm):
    trust = make_text_pdf(settings.data_dir / "trust.pdf", TRUST_PAGES)
    build_index(vsm, settings)

    (settings.data_dir / "trust_copy.pdf").write_bytes(trust.read_bytes())
    stats = build_index(vsm, settings)

    assert "trust_copy.pdf" in stats["skipped_duplicates"]
    assert vsm.count() == 2


def test_unreadable_pdf_keeps_its_existing_chunks(settings, vsm):
    trust = make_text_pdf(settings.data_dir / "trust.pdf", TRUST_PAGES)
    make_text_pdf(settings.data_dir / "fees.pdf", FEES_PAGES)
    build_index(vsm, settings)

    trust.write_bytes(b"not a pdf any more")
    stats = build_index(vsm, settings)

    assert "trust.pdf" in stats["failed_files"]
    assert vsm.indexed_sources() == {"trust.pdf", "fees.pdf"}


def test_empty_directory_does_not_wipe_existing_index(settings, vsm):
    trust = make_text_pdf(settings.data_dir / "trust.pdf", TRUST_PAGES)
    build_index(vsm, settings)
    trust.unlink()

    with pytest.raises(IndexingError, match="left unchanged"):
        build_index(vsm, settings)
    with pytest.raises(IndexingError, match="left unchanged"):
        build_index(vsm, settings, rebuild=True)

    assert vsm.count() == 2


def test_empty_directory_with_empty_index_is_a_noop(settings, vsm):
    stats = build_index(vsm, settings)
    assert stats["chunks_added"] == 0
    assert vsm.count() == 0


def test_rebuild_replaces_index_and_leaves_no_staging_collection(settings, vsm):
    trust = make_text_pdf(settings.data_dir / "trust.pdf", TRUST_PAGES)
    build_index(vsm, settings)
    trust.unlink()
    make_text_pdf(settings.data_dir / "fees.pdf", FEES_PAGES)

    stats = build_index(vsm, settings, rebuild=True)

    assert stats["chunks_added"] == 1
    assert vsm.indexed_sources() == {"fees.pdf"}
    names = {c.name for c in vsm.get_store()._client.list_collections()}
    assert names == {settings.collection_name}


def test_failed_rebuild_keeps_existing_index(settings, vsm, embeddings):
    make_text_pdf(settings.data_dir / "trust.pdf", TRUST_PAGES)
    build_index(vsm, settings)
    make_text_pdf(settings.data_dir / "fees.pdf", FEES_PAGES)

    with patch.object(embeddings, "embed_documents", side_effect=RuntimeError("model crashed")):
        with pytest.raises(IndexingError, match="model crashed.*left unchanged"):
            build_index(vsm, settings, rebuild=True)

    assert vsm.indexed_sources() == {"trust.pdf"}
    assert vsm.count() == 2
    names = {c.name for c in vsm.get_store()._client.list_collections()}
    assert names == {settings.collection_name}


def test_failed_incremental_index_keeps_existing_chunks(settings, vsm, embeddings):
    pdf = make_text_pdf(settings.data_dir / "trust.pdf", TRUST_PAGES)
    build_index(vsm, settings)
    make_text_pdf(pdf, ["Completely new first page.", "Completely new second page."])

    with patch.object(embeddings, "embed_documents", side_effect=RuntimeError("model crashed")):
        with pytest.raises(IndexingError, match="left in place"):
            build_index(vsm, settings)

    assert any("Sixty two percent" in t for t in _texts_for(vsm, "trust.pdf"))


def test_large_batches_are_split_within_chroma_limits(settings, vsm):
    documents = [
        Document(
            page_content=f"chunk number {i}",
            metadata={"source": "big.pdf", "page": i // 4 + 1, "chunk_index": i % 4, "chunk_hash": f"h{i}", "file_hash": "f"},
        )
        for i in range(6000)  # above Chroma's 5461 maximum batch size
    ]

    result = vsm.sync_documents(documents, current_files={"big.pdf": "f"})
    assert result.added == 6000
    assert vsm.count() == 6000

    result = vsm.sync_documents(documents[:100], current_files={"big.pdf": "f"})
    assert result.removed == 5900
    assert vsm.count() == 100
