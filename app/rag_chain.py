"""
End-to-end RAG pipeline:

  User Question -> SemanticRetriever -> ChromaDB -> relevant chunks
    -> grounded prompt -> Mistral (via Ollama) -> answer + citations

The LLM is injected so tests can supply a mock and run without a live
Ollama server.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

from app.config import Settings, settings as default_settings
from app.prompts import build_rag_prompt
from app.retriever import RetrievedPassage, SemanticRetriever
from app.utils import get_logger, truncate_chars

logger = get_logger(__name__)


class SupportsInvoke(Protocol):
    def invoke(self, prompt: str): ...


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: List[RetrievedPassage]
    context_used: str


def format_context(passages: List[RetrievedPassage], max_chars: int) -> str:
    """Render retrieved passages into a citation-friendly context block,
    truncated to stay well within the local model's context window."""
    if not passages:
        return "(No relevant passages were found in the indexed research corpus.)"

    blocks = []
    for i, p in enumerate(passages, start=1):
        blocks.append(
            f"[Passage {i}] Source: {p.source}, Page: {p.page} (relevance {p.relevance_score:.2f})\n{p.text}"
        )
    context = "\n\n".join(blocks)
    return truncate_chars(context, max_chars)


def _extract_text(response) -> str:
    """LangChain chat models return an AIMessage; plain callables/mocks may
    return a string directly. Support both."""
    if hasattr(response, "content"):
        return response.content
    return str(response)


class RAGChain:
    """Retrieval-augmented generation over the local research corpus."""

    def __init__(
        self,
        retriever: Optional[SemanticRetriever] = None,
        llm: Optional[SupportsInvoke] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or default_settings
        self.retriever = retriever or SemanticRetriever(settings=self.settings)
        self._llm = llm  # lazily built from app.llm.get_chat_model if not provided

    def _get_llm(self) -> SupportsInvoke:
        if self._llm is None:
            from app.llm import get_chat_model  # local import avoids hard dependency at module load

            self._llm = get_chat_model(self.settings)
        return self._llm

    def answer(self, question: str, k: Optional[int] = None) -> RAGResult:
        passages = self.retriever.search(question, k=k)
        context = format_context(passages, self.settings.max_context_chars)
        prompt = build_rag_prompt(question=question, context=context)

        llm = self._get_llm()
        try:
            raw_response = llm.invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM generation failed: %s", exc)
            return RAGResult(
                question=question,
                answer=(
                    "The local Mistral model could not be reached. Please confirm Ollama is running "
                    "('ollama list' to check installed models, 'ollama run mistral' to test) and try again."
                ),
                sources=passages,
                context_used=context,
            )

        answer_text = _extract_text(raw_response)
        return RAGResult(question=question, answer=answer_text, sources=passages, context_used=context)
