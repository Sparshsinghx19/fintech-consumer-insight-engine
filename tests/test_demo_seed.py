"""Tests for seeding the public demo index from the bundled demo corpus,
using an isolated Chroma collection in a temporary directory and the real
demo_data PDFs."""
from __future__ import annotations

import re
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from pypdf import PdfReader

from app.config import DEMO_DATA_DIR, Settings
from app.indexing import SEED_ALREADY_INDEXED, SEED_DISABLED, SEED_SEEDED, seed_demo_index
from app.vector_store import IndexingError, VectorStoreManager
from tests.helpers import HashingEmbeddings

DEMO_DIR = Path(__file__).resolve().parent.parent / DEMO_DATA_DIR
DEMO_PDFS = sorted(DEMO_DIR.glob("*.pdf"))


@pytest.fixture
def demo_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=DEMO_DIR,
        processed_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vs",
        public_demo_mode=True,
    )


@pytest.fixture
def vsm(demo_settings: Settings) -> VectorStoreManager:
    with patch("app.vector_store.get_embedding_function", return_value=HashingEmbeddings()):
        return VectorStoreManager(demo_settings)


# -- the bundled corpus ------------------------------------------------------


def test_demo_corpus_is_two_labelled_text_pdfs():
    assert [p.name for p in DEMO_PDFS] == [
        "consumer_trust_digital_payments_demo_report.pdf",
        "digital_payments_adoption_demo_report.pdf",
    ]
    for pdf in DEMO_PDFS:
        reader = PdfReader(pdf)
        assert "Demo Research Report" in reader.metadata.title
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            assert "Illustrative demo content" in text and "Not empirical research" in text
            # No statistics: the only digits are the page footer.
            assert re.findall(r"\d+", text.replace(f"Page {number} of {len(reader.pages)}", "")) == []


# -- seeding -----------------------------------------------------------------


def test_seeds_empty_index_with_page_metadata(vsm, demo_settings):
    result = seed_demo_index(vsm, demo_settings)

    assert result.status == SEED_SEEDED
    assert set(result.documents) == {p.name for p in DEMO_PDFS}
    assert result.chunks_added == vsm.count() > 0
    stats = vsm.source_stats()
    assert set(stats) == {p.name for p in DEMO_PDFS}
    assert all(s.pages == 5 for s in stats.values())
    hits = vsm.get_store().similarity_search("hidden fees and merchant acceptance barriers", k=3)
    assert any(h.metadata["source"] == "digital_payments_adoption_demo_report.pdf" for h in hits)


def test_seeding_is_idempotent(vsm, demo_settings):
    first = seed_demo_index(vsm, demo_settings)
    count = vsm.count()

    second = seed_demo_index(vsm, demo_settings)

    assert first.status == SEED_SEEDED and second.status == SEED_ALREADY_INDEXED
    assert second.chunks_added == 0 and vsm.count() == count


def test_existing_index_is_never_modified(vsm, demo_settings):
    existing = Document(
        page_content="A researcher's own document.",
        metadata={"source": "my_research.pdf", "page": 1, "chunk_hash": "own-1", "file_hash": "h"},
    )
    vsm.add_documents([existing])

    result = seed_demo_index(vsm, demo_settings)

    assert result.status == SEED_ALREADY_INDEXED
    assert vsm.count() == 1 and vsm.indexed_sources() == {"my_research.pdf"}


def test_nothing_happens_outside_demo_mode(vsm, demo_settings):
    result = seed_demo_index(vsm, replace(demo_settings, public_demo_mode=False))
    assert result.status == SEED_DISABLED and vsm.count() == 0


def test_concurrent_seeding_indexes_once(vsm, demo_settings):
    results = []
    threads = [threading.Thread(target=lambda: results.append(seed_demo_index(vsm, demo_settings))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(r.status for r in results) == [SEED_ALREADY_INDEXED] * 3 + [SEED_SEEDED]
    seeded = next(r for r in results if r.status == SEED_SEEDED)
    assert vsm.count() == seeded.chunks_added


def test_missing_demo_documents_raise_a_clear_error(vsm, demo_settings, tmp_path):
    empty = tmp_path / "empty_demo"
    empty.mkdir()
    with pytest.raises(IndexingError, match="No demo documents"):
        seed_demo_index(vsm, replace(demo_settings, data_dir=empty))
    assert vsm.count() == 0


# -- data directory defaults ---------------------------------------------------


def test_demo_mode_defaults_to_bundled_corpus(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "true")
    assert Settings().data_dir == Path(DEMO_DATA_DIR)

    monkeypatch.setenv("DATA_DIR", "my/research")
    assert Settings().data_dir == Path("my/research")  # explicit setting wins


def test_local_development_keeps_data_raw(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("PUBLIC_DEMO_MODE", raising=False)
    assert Settings().data_dir == Path("data/raw")
