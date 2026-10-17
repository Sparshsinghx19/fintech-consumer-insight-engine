"""
Session-scoped research history and saved queries for the Research
Assistant.

Plain in-memory objects with no Streamlit dependency: the UI keeps one
instance of each per browser session in `st.session_state`, so nothing is
shared between users, written to disk, or kept after the session ends.
Both collections are bounded to limit memory use.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from app.rag_chain import RAGResult, passages_in_context
from app.retriever import RetrievedPassage

MAX_HISTORY_ENTRIES = 20
MAX_SAVED_QUERIES = 30


def normalize_question(question: str) -> str:
    """Comparison key for questions: case- and whitespace-insensitive."""
    return " ".join(question.split()).casefold()


@dataclass
class ResearchEntry:
    """One answered (or failed) research question with the exact evidence
    that was retrieved for it."""

    question: str
    answer: str
    sources: List[RetrievedPassage]
    in_context: List[bool]
    k: int
    provider: str
    model: str
    error: Optional[str] = None
    asked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def entry_from_result(result: RAGResult, k: int, provider: str, model: str) -> ResearchEntry:
    return ResearchEntry(
        question=result.question,
        answer=result.answer,
        sources=list(result.sources),
        in_context=passages_in_context(result),
        k=k,
        provider=provider,
        model=model,
        error=result.error,
    )


class ResearchHistory:
    """Most recent research entries, newest first, capped at `max_entries`."""

    def __init__(self, max_entries: int = MAX_HISTORY_ENTRIES):
        self.max_entries = max_entries
        self._entries: List[ResearchEntry] = []

    def add(self, entry: ResearchEntry) -> None:
        self._entries.insert(0, entry)
        del self._entries[self.max_entries :]

    def get(self, entry_id: Optional[str]) -> Optional[ResearchEntry]:
        return next((e for e in self._entries if e.id == entry_id), None)

    def latest_for(self, question: str) -> Optional[ResearchEntry]:
        key = normalize_question(question)
        return next((e for e in self._entries if normalize_question(e.question) == key), None)

    def clear(self) -> None:
        self._entries.clear()

    @property
    def entries(self) -> List[ResearchEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class SavedQuery:
    question: str
    k: int
    saved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SavedQueries:
    """Questions the user chose to keep, independent of history (removing
    an answer from history never removes a saved query, and vice versa).
    Newest first, de-duplicated by question, capped at `max_items`."""

    def __init__(self, max_items: int = MAX_SAVED_QUERIES):
        self.max_items = max_items
        self._items: List[SavedQuery] = []

    def is_saved(self, question: str) -> bool:
        key = normalize_question(question)
        return any(normalize_question(q.question) == key for q in self._items)

    def save(self, question: str, k: int) -> bool:
        """Save a question; returns False if it was already saved or the
        list is full."""
        if not question.strip() or self.is_saved(question) or len(self._items) >= self.max_items:
            return False
        self._items.insert(0, SavedQuery(question=" ".join(question.split()), k=k))
        return True

    def unsave(self, question: str) -> None:
        key = normalize_question(question)
        self._items = [q for q in self._items if normalize_question(q.question) != key]

    @property
    def items(self) -> List[SavedQuery]:
        return list(self._items)

    @property
    def is_full(self) -> bool:
        return len(self._items) >= self.max_items

    def __len__(self) -> int:
        return len(self._items)


def answer_markdown(entry: ResearchEntry) -> str:
    """The answer as a Markdown document (for copy/download), with the
    passages that were actually retrieved listed after it."""
    lines = [f"# {entry.question}", "", entry.answer.strip(), ""]
    if entry.sources:
        lines += ["## Retrieved passages", ""]
        for i, (p, used) in enumerate(zip(entry.sources, entry.in_context), start=1):
            note = "" if used else " (not sent to the model: context limit)"
            lines.append(f"{i}. {p.source}, p. {p.page} — similarity {p.relevance_score:.2f}{note}")
        lines.append("")
    lines.append(f"_Generated {entry.asked_at:%Y-%m-%d %H:%M UTC} by {entry.model} ({entry.provider})._")
    return "\n".join(lines)
