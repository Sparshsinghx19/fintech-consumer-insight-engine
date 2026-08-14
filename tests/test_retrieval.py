"""Tests for semantic retrieval, using a mocked Chroma store so no real
embedding model or persisted database is required."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.retriever import SemanticRetriever
from app.vector_store import VectorStoreManager


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vs",
        retrieval_k=3,
    )


def _mock_vsm(settings: Settings, search_results):
    vsm = MagicMock(spec=VectorStoreManager)
    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = search_results
    vsm.get_store.return_value = mock_store
    return vsm


def test_search_returns_passages_with_metadata(settings: Settings):
    docs_and_scores = [
        (Document(page_content="Trust drives adoption.", metadata={"source": "trust.pdf", "page": 3, "chunk_hash": "h1"}), 0.2),
        (Document(page_content="Fees are a barrier.", metadata={"source": "fees.pdf", "page": 9, "chunk_hash": "h2"}), 0.5),
    ]
    vsm = _mock_vsm(settings, docs_and_scores)
    retriever = SemanticRetriever(vector_store_manager=vsm, settings=settings)

    results = retriever.search("What drives trust?", k=2)

    assert len(results) == 2
    assert results[0].source == "trust.pdf"
    assert results[0].page == 3
    assert 0.0 <= results[0].relevance_score <= 1.0
    assert results[1].source == "fees.pdf"


def test_search_empty_query_returns_empty_list(settings: Settings):
    vsm = _mock_vsm(settings, [])
    retriever = SemanticRetriever(vector_store_manager=vsm, settings=settings)
    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_search_handles_store_exception_gracefully(settings: Settings):
    vsm = MagicMock(spec=VectorStoreManager)
    mock_store = MagicMock()
    mock_store.similarity_search_with_score.side_effect = RuntimeError("boom")
    vsm.get_store.return_value = mock_store
    retriever = SemanticRetriever(vector_store_manager=vsm, settings=settings)

    results = retriever.search("anything")
    assert results == []


def test_search_uses_default_k_from_settings(settings: Settings):
    vsm = _mock_vsm(settings, [])
    retriever = SemanticRetriever(vector_store_manager=vsm, settings=settings)
    retriever.search("query without explicit k")
    called_kwargs = vsm.get_store.return_value.similarity_search_with_score.call_args
    assert called_kwargs.kwargs.get("k") == settings.retrieval_k
