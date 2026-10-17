"""Tests for VectorStoreManager initialization and incremental-indexing
duplicate detection, using a mocked embedding function so no model
weights need to be downloaded during unit testing."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.vector_store import IndexingError, VectorStoreManager


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "raw", processed_dir=tmp_path / "processed", vectorstore_dir=tmp_path / "vs")


@pytest.fixture
def fake_embeddings():
    fake = MagicMock()
    fake.embed_documents.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    fake.embed_query.return_value = [0.1, 0.2, 0.3]
    return fake


def test_vector_store_manager_creates_persist_directory(settings: Settings, fake_embeddings):
    with patch("app.vector_store.get_embedding_function", return_value=fake_embeddings):
        VectorStoreManager(settings)
    assert settings.vectorstore_dir.exists()


def test_store_persists_to_disk_with_telemetry_disabled(settings: Settings, fake_embeddings):
    with patch("app.vector_store.get_embedding_function", return_value=fake_embeddings):
        vsm = VectorStoreManager(settings)
        vsm.add_documents(
            [Document(page_content="persisted chunk", metadata={"chunk_hash": "p1", "source": "a.pdf", "page": 1})]
        )
        client_settings = vsm.get_store()._client_settings
        assert client_settings.anonymized_telemetry is False
        assert client_settings.is_persistent is True

    assert (settings.vectorstore_dir / "chroma.sqlite3").exists()


def test_add_documents_skips_existing_chunk_hashes(settings: Settings, fake_embeddings):
    with patch("app.vector_store.get_embedding_function", return_value=fake_embeddings):
        vsm = VectorStoreManager(settings)

        vsm.existing_chunk_hashes = MagicMock(return_value={"hash_1"})
        mock_store = MagicMock()
        vsm.get_store = MagicMock(return_value=mock_store)

        docs = [
            Document(page_content="new content", metadata={"chunk_hash": "hash_2", "source": "a.pdf", "page": 1}),
            Document(page_content="already indexed", metadata={"chunk_hash": "hash_1", "source": "a.pdf", "page": 2}),
        ]
        added = vsm.add_documents(docs, skip_duplicates=True)

        assert added == 1
        called_docs = mock_store.add_documents.call_args.kwargs["documents"]
        assert len(called_docs) == 1
        assert called_docs[0].metadata["chunk_hash"] == "hash_2"


def test_add_documents_with_empty_list_is_noop(settings: Settings, fake_embeddings):
    with patch("app.vector_store.get_embedding_function", return_value=fake_embeddings):
        vsm = VectorStoreManager(settings)
        vsm.get_store = MagicMock()
        added = vsm.add_documents([])
        assert added == 0
        vsm.get_store.return_value.add_documents.assert_not_called()


def test_delete_by_source_removes_matching_chunks(settings: Settings, fake_embeddings):
    with patch("app.vector_store.get_embedding_function", return_value=fake_embeddings):
        vsm = VectorStoreManager(settings)
        mock_store = MagicMock()
        mock_store.get.return_value = {"ids": ["h1", "h2"], "metadatas": [{"source": "a.pdf"}, {"source": "a.pdf"}]}
        vsm.get_store = MagicMock(return_value=mock_store)

        removed = vsm.delete_by_source("a.pdf")

        assert removed == 2
        mock_store.get.assert_called_once_with(where={"source": "a.pdf"}, include=["metadatas"])
        mock_store.delete.assert_called_once_with(ids=["h1", "h2"])


def test_delete_by_source_with_no_matches_is_noop(settings: Settings, fake_embeddings):
    with patch("app.vector_store.get_embedding_function", return_value=fake_embeddings):
        vsm = VectorStoreManager(settings)
        mock_store = MagicMock()
        mock_store.get.return_value = {"ids": [], "metadatas": []}
        vsm.get_store = MagicMock(return_value=mock_store)

        removed = vsm.delete_by_source("missing.pdf")

        assert removed == 0
        mock_store.delete.assert_not_called()


def test_rebuild_with_no_documents_refuses_and_touches_nothing(settings: Settings, fake_embeddings):
    with patch("app.vector_store.get_embedding_function", return_value=fake_embeddings):
        vsm = VectorStoreManager(settings)
        vsm.reset = MagicMock()
        vsm.get_store = MagicMock()

        with pytest.raises(IndexingError, match="left unchanged"):
            vsm.rebuild([])

        vsm.reset.assert_not_called()
        vsm.get_store.assert_not_called()
