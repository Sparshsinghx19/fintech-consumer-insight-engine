"""Tests for session-scoped research history, saved queries, and the
evidence helpers used by the Research Assistant workspace."""
from __future__ import annotations

from app.rag_chain import RAGResult
from app.research_session import (
    ResearchEntry,
    ResearchHistory,
    SavedQueries,
    answer_markdown,
    entry_from_result,
    normalize_question,
)
from app.retriever import RetrievedPassage
from app.ui import evidence_by_document, evidence_table_markdown, truncate


def _passage(source: str, page: int, score: float) -> RetrievedPassage:
    return RetrievedPassage(text=f"{source} p{page}", source=source, page=page, document_type="pdf", relevance_score=score)


def _entry(question: str, **kwargs) -> ResearchEntry:
    defaults = dict(answer="A", sources=[], in_context=[], k=5, provider="Ollama (local)", model="mistral")
    return ResearchEntry(question=question, **{**defaults, **kwargs})


def test_history_is_newest_first_and_bounded():
    history = ResearchHistory(max_entries=3)
    entries = [_entry(f"q{i}") for i in range(5)]
    for entry in entries:
        history.add(entry)

    assert [e.question for e in history.entries] == ["q4", "q3", "q2"]
    assert history.get(entries[0].id) is None  # trimmed
    assert history.get(entries[4].id) is entries[4]


def test_history_keeps_each_answer_with_its_own_evidence():
    history = ResearchHistory()
    first = _entry("fees?", sources=[_passage("fees.pdf", 3, 0.8)], in_context=[True])
    second = _entry("trust?", sources=[_passage("trust.pdf", 1, 0.7)], in_context=[True])
    history.add(first)
    history.add(second)

    assert history.get(first.id).sources[0].source == "fees.pdf"
    assert history.get(second.id).sources[0].source == "trust.pdf"


def test_latest_for_matches_normalized_question():
    history = ResearchHistory()
    old, new = _entry("What drives adoption?"), _entry("what  drives ADOPTION?")
    history.add(old)
    history.add(new)
    assert history.latest_for("What drives adoption?") is new
    assert history.latest_for("unrelated") is None


def test_clear_history_does_not_touch_saved_queries():
    history, saved = ResearchHistory(), SavedQueries()
    history.add(_entry("q"))
    saved.save("q", 5)
    history.clear()
    assert len(history) == 0 and saved.is_saved("q")


def test_saved_queries_dedupe_unsave_and_limit():
    saved = SavedQueries(max_items=2)
    assert saved.save("What drives adoption?", 5)
    assert not saved.save("  what drives   adoption? ", 8)  # duplicate
    assert saved.save("Fees?", 3)
    assert saved.is_full and not saved.save("Third?", 5)
    assert [q.question for q in saved.items] == ["Fees?", "What drives adoption?"]

    saved.unsave("WHAT DRIVES ADOPTION?")
    assert not saved.is_saved("What drives adoption?") and len(saved) == 1
    assert not saved.save("   ", 5)


def test_entry_from_result_records_context_flags_and_error():
    sources = [_passage("a.pdf", 1, 0.9), _passage("b.pdf", 2, 0.5)]
    result = RAGResult(question="q", answer="msg", sources=sources, context_used="[Passage 1] Source: a.pdf", error="msg")

    entry = entry_from_result(result, k=2, provider="Gemini API (cloud)", model="gemini-3.8-flash")

    assert entry.in_context == [True, False]
    assert entry.error == "msg" and entry.k == 2 and entry.model == "gemini-3.8-flash"
    assert entry.sources == sources and entry.sources is not sources


def test_answer_markdown_lists_retrieved_passages():
    entry = _entry(
        "Fees?", answer="Fees matter (fees.pdf, p. 3).",
        sources=[_passage("fees.pdf", 3, 0.81), _passage("x.pdf", 9, 0.2)], in_context=[True, False],
    )
    md = answer_markdown(entry)
    assert md.startswith("# Fees?")
    assert "1. fees.pdf, p. 3 — similarity 0.81" in md
    assert "2. x.pdf, p. 9 — similarity 0.20 (not sent to the model: context limit)" in md


def test_evidence_by_document_groups_real_scores():
    rows = evidence_by_document(
        [_passage("a.pdf", 2, 0.5), _passage("b.pdf", 1, 0.9), _passage("a.pdf", 1, 0.7)], [True, True, False]
    )
    assert rows == [
        {"Document": "b.pdf", "Pages": "1", "Passages": 1, "Sent to model": 1, "Best similarity": 0.9},
        {"Document": "a.pdf", "Pages": "1, 2", "Passages": 2, "Sent to model": 1, "Best similarity": 0.7},
    ]


def test_evidence_table_markdown():
    rows = evidence_by_document([_passage("a|b.pdf", 1, 0.5)], [True])
    assert evidence_table_markdown(rows).splitlines() == [
        "| Document | Pages | Passages | Sent to model | Best similarity |",
        "|---|---|---|---|---|",
        "| a\\|b.pdf | 1 | 1 | 1 | 0.50 |",
    ]
    assert evidence_table_markdown([]) == ""


def test_helpers():
    assert normalize_question("  A  b ") == "a b"
    assert truncate("one two three four", 10) == "one two…"
    assert truncate("short", 10) == "short"
