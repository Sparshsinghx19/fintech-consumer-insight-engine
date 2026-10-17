"""Tests for the backend hooks and pure helpers used by the redesigned UI:
document status comparison, per-source index stats, answer splitting,
context-inclusion flags, report progress/errors, and error fields."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.indexing import (
    CHANGED,
    DUPLICATE,
    INDEXED,
    NO_TEXT,
    NOT_INDEXED,
    REMOVED,
    UNREADABLE,
    document_statuses,
)
from app.insights import UNSTRUCTURED_PLACEHOLDER, InsightEngine, parse_insight_response
from app.rag_chain import RAGChain, passages_in_context
from app.reporting import REPORT_LLM_STEPS, ReportGenerator
from app.retriever import RetrievedPassage, SemanticRetriever
from app.ui import report_preview_markdown, split_model_answer
from app.vector_store import SourceStats, VectorStoreManager
from tests.helpers import HashingEmbeddings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "raw", processed_dir=tmp_path / "processed", vectorstore_dir=tmp_path / "vs")


def _passage(i: int, text: str = "Evidence text.") -> RetrievedPassage:
    return RetrievedPassage(text=text, source=f"doc{i}.pdf", page=i, document_type="pdf", relevance_score=0.8)


# -- document statuses ----------------------------------------------------


def test_document_statuses_cover_every_state(tmp_path: Path):
    names = ["same.pdf", "edited.pdf", "new.pdf", "copy.pdf", "scan.pdf", "locked.pdf"]
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    hashes = {"same.pdf": "h-same", "edited.pdf": "h-edited-v2", "new.pdf": "h-new", "copy.pdf": "h-same", "scan.pdf": "h-scan", "locked.pdf": None}
    indexed = {
        "same.pdf": SourceStats(chunks=4, pages=2, file_hash="h-same"),
        "edited.pdf": SourceStats(chunks=3, pages=1, file_hash="h-edited-v1"),
        "gone.pdf": SourceStats(chunks=5, pages=3, file_hash="h-gone"),
    }

    statuses = document_statuses(
        [tmp_path / n for n in names], lambda p: hashes[p.name], indexed, no_text_files=["scan.pdf"]
    )
    by_name = {s.name: s for s in statuses}

    assert by_name["same.pdf"].status == INDEXED and by_name["same.pdf"].chunks == 4
    assert by_name["edited.pdf"].status == CHANGED
    assert by_name["new.pdf"].status == NOT_INDEXED
    assert by_name["copy.pdf"].status == DUPLICATE and by_name["copy.pdf"].duplicate_of == "same.pdf"
    assert by_name["scan.pdf"].status == NO_TEXT
    assert by_name["locked.pdf"].status == UNREADABLE
    assert by_name["gone.pdf"].status == REMOVED and by_name["gone.pdf"].size_bytes is None
    assert {s.name for s in statuses if s.needs_indexing} == {"edited.pdf", "new.pdf", "gone.pdf"}


def test_renamed_file_is_not_reported_as_duplicate_of_its_old_name(tmp_path: Path):
    (tmp_path / "renamed.pdf").write_bytes(b"x")
    indexed = {"original.pdf": SourceStats(chunks=2, pages=2, file_hash="h1")}

    statuses = {s.name: s.status for s in document_statuses([tmp_path / "renamed.pdf"], lambda p: "h1", indexed)}

    assert statuses == {"renamed.pdf": NOT_INDEXED, "original.pdf": REMOVED}


def test_source_stats_aggregates_chunks_and_pages(settings: Settings):
    with patch("app.vector_store.get_embedding_function", return_value=HashingEmbeddings()):
        vsm = VectorStoreManager(settings)
        docs = [
            Document(page_content=f"text {i}", metadata={"source": "a.pdf", "page": i // 2 + 1, "chunk_hash": f"a{i}", "file_hash": "ha"})
            for i in range(5)
        ] + [Document(page_content="b", metadata={"source": "b.pdf", "page": 1, "chunk_hash": "b0", "file_hash": "hb"})]
        vsm.add_documents(docs)

        stats = vsm.source_stats()

    assert stats == {"a.pdf": SourceStats(chunks=5, pages=3, file_hash="ha"), "b.pdf": SourceStats(chunks=1, pages=1, file_hash="hb")}


# -- research assistant helpers --------------------------------------------


def test_split_model_answer_separates_cited_sources():
    body, cited = split_model_answer("ANSWER:\nFees deter users (fees.pdf, p. 3).\n\nSOURCES USED:\n- fees.pdf, Page: 3")
    assert body == "Fees deter users (fees.pdf, p. 3)."
    assert cited == "- fees.pdf, Page: 3"


def test_split_model_answer_leaves_unstructured_text_alone():
    assert split_model_answer("Just an answer.") == ("Just an answer.", "")


def test_passages_in_context_flags_truncated_passages(settings: Settings):
    passages = [_passage(i, "x" * 900) for i in range(1, 6)]
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.return_value = passages
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="ok")
    small = Settings(**{**settings.__dict__, "max_context_chars": 2000})

    result = RAGChain(retriever=retriever, llm=llm, settings=small).answer("q")

    flags = passages_in_context(result)
    assert flags[:2] == [True, True]
    assert flags[-1] is False
    assert result.error is None


def test_rag_result_sets_error_on_failure(settings: Settings):
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.return_value = []
    llm = MagicMock()
    llm.invoke.side_effect = ConnectionError("down")

    result = RAGChain(retriever=retriever, llm=llm, settings=settings).answer("q")

    assert result.error and result.error == result.answer


# -- insights and reports ----------------------------------------------------


def test_insight_error_field_set_only_on_failure(settings: Settings):
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.return_value = []
    failing = MagicMock()
    failing.invoke.side_effect = ConnectionError("down")
    assert InsightEngine(retriever=retriever, llm=failing, settings=settings).generate_insight("t").error

    assert parse_insight_response("INSIGHT_TITLE: T\nEVIDENCE: E", topic="t").error == ""
    assert parse_insight_response("free text", topic="t").consumer_problem == UNSTRUCTURED_PLACEHOLDER


def test_report_progress_and_errors(settings: Settings):
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.return_value = [_passage(1)]
    llm = MagicMock()
    llm.invoke.side_effect = [MagicMock(content="ok")] * 2 + [ConnectionError("down")] + [MagicMock(content="ok")] * 20
    calls = []

    report = ReportGenerator(retriever=retriever, llm=llm, settings=settings).generate_report(
        "wallets", progress_callback=lambda done, total, label: calls.append((done, total, label))
    )

    assert [c[0] for c in calls] == list(range(REPORT_LLM_STEPS + 1))
    assert all(c[1] == REPORT_LLM_STEPS for c in calls)
    assert calls[0][2] == "Executive Summary" and calls[-1][2] == "Done"
    assert len(report.errors) == 1 and report.errors[0].startswith("Consumer Pain Points:")


def test_report_without_progress_callback_still_works(settings: Settings):
    retriever = MagicMock(spec=SemanticRetriever)
    retriever.search.return_value = []
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="ok")
    report = ReportGenerator(retriever=retriever, llm=llm, settings=settings).generate_report("t")
    assert report.errors == [] and report.markdown.startswith("# FinTech Consumer Insight Report")


def test_report_preview_demotes_headings_only():
    md = "# Title\n## Sub\nText with # inside\n"
    assert report_preview_markdown(md) == "## Title\n### Sub\nText with # inside\n"
