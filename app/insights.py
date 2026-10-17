"""
Consumer Insight Engine.

Generates structured, evidence-grounded consumer insights from the
research corpus. Every insight is parsed into a Pydantic model that
keeps EVIDENCE, INTERPRETATION, and RECOMMENDATION cleanly separated so
recommendations are never presented as research findings.
"""
from __future__ import annotations

import re
from typing import List, Optional, Protocol

from pydantic import BaseModel, Field

from app.config import Settings, settings as default_settings
from app.prompts import build_insight_prompt
from app.rag_chain import format_context
from app.retriever import SemanticRetriever
from app.utils import get_logger

logger = get_logger(__name__)


class SupportsInvoke(Protocol):
    def invoke(self, prompt: str): ...


class ConsumerInsight(BaseModel):
    """A single structured, evidence-grounded consumer insight."""

    title: str = Field(..., description="Short, specific insight title")
    consumer_problem: str = Field(..., description="The consumer behavior or problem this insight addresses")
    evidence: str = Field(..., description="What the retrieved research directly supports")
    interpretation: str = Field(..., description="What can reasonably be inferred from the evidence")
    sources: str = Field(default="", description="Document + page citations actually used")
    confidence: str = Field(default="Medium", description="High, Medium, or Low")
    business_implication: str = Field(default="")
    recommended_action: str = Field(default="", description="A recommendation - never presented as a finding")
    potential_opportunity: str = Field(default="")
    topic: str = Field(default="", description="The research topic/question this insight was generated for")
    error: str = Field(default="", description="Set when generation failed; the other fields are then placeholders")


# consumer_problem value used when the model ignored the structured format
# and the raw response was stored as evidence.
UNSTRUCTURED_PLACEHOLDER = "Not explicitly structured by the model."


_FIELD_PATTERNS = {
    "title": r"INSIGHT_TITLE:\s*(.*)",
    "consumer_problem": r"CONSUMER_PROBLEM:\s*(.*)",
    "evidence": r"EVIDENCE:\s*(.*)",
    "interpretation": r"INTERPRETATION:\s*(.*)",
    "sources": r"SOURCES:\s*(.*)",
    "confidence": r"CONFIDENCE:\s*(.*)",
    "business_implication": r"BUSINESS_IMPLICATION:\s*(.*)",
    "recommended_action": r"RECOMMENDED_ACTION:\s*(.*)",
    "potential_opportunity": r"POTENTIAL_OPPORTUNITY:\s*(.*)",
}

_LABELS = list(_FIELD_PATTERNS.keys())
_LABEL_KEYS = {
    "title": "INSIGHT_TITLE",
    "consumer_problem": "CONSUMER_PROBLEM",
    "evidence": "EVIDENCE",
    "interpretation": "INTERPRETATION",
    "sources": "SOURCES",
    "confidence": "CONFIDENCE",
    "business_implication": "BUSINESS_IMPLICATION",
    "recommended_action": "RECOMMENDED_ACTION",
    "potential_opportunity": "POTENTIAL_OPPORTUNITY",
}


def parse_insight_response(raw_text: str, topic: str) -> ConsumerInsight:
    """Parse the model's structured plain-text response into a
    ConsumerInsight. Falls back to placing the whole response in
    `evidence` if the model did not follow the requested format, so a
    malformed response never crashes the app."""
    values: dict[str, str] = {}
    keys = list(_LABEL_KEYS.items())

    for i, (field_name, label) in enumerate(keys):
        start_match = re.search(rf"{label}:\s*", raw_text)
        if not start_match:
            continue
        start = start_match.end()
        # find the next label's position to know where this field ends
        end = len(raw_text)
        for other_field, other_label in keys:
            if other_field == field_name:
                continue
            m = re.search(rf"\n{other_label}:", raw_text[start:])
            if m:
                candidate_end = start + m.start()
                if candidate_end < end:
                    end = candidate_end
        values[field_name] = raw_text[start:end].strip()

    if not values.get("title") and not values.get("evidence"):
        logger.warning("Model response did not follow the expected insight format; storing raw text as evidence.")
        return ConsumerInsight(
            title=f"Insight on {topic}",
            consumer_problem=UNSTRUCTURED_PLACEHOLDER,
            evidence=raw_text.strip(),
            interpretation="",
            topic=topic,
        )

    return ConsumerInsight(
        title=values.get("title") or f"Insight on {topic}",
        consumer_problem=values.get("consumer_problem", ""),
        evidence=values.get("evidence", ""),
        interpretation=values.get("interpretation", ""),
        sources=values.get("sources", ""),
        confidence=values.get("confidence", "Medium") or "Medium",
        business_implication=values.get("business_implication", ""),
        recommended_action=values.get("recommended_action", ""),
        potential_opportunity=values.get("potential_opportunity", ""),
        topic=topic,
    )


def _extract_text(response) -> str:
    if hasattr(response, "content"):
        return response.content
    return str(response)


class InsightEngine:
    """Generates structured consumer insights for one or more topics."""

    def __init__(
        self,
        retriever: Optional[SemanticRetriever] = None,
        llm: Optional[SupportsInvoke] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or default_settings
        self.retriever = retriever or SemanticRetriever(settings=self.settings)
        self._llm = llm

    def _get_llm(self) -> SupportsInvoke:
        if self._llm is None:
            from app.llm import get_chat_model

            self._llm = get_chat_model(self.settings)
        return self._llm

    def generate_insight(self, topic: str, k: Optional[int] = None) -> ConsumerInsight:
        passages = self.retriever.search(topic, k=k)
        context = format_context(passages, self.settings.max_context_chars)
        prompt = build_insight_prompt(topic=topic, context=context)

        try:
            raw_response = self._get_llm().invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            from app.llm import describe_llm_error

            logger.error("Insight generation failed for topic '%s': %s", topic, exc)
            message = describe_llm_error(exc, self.settings)
            return ConsumerInsight(
                title=f"Could not generate insight on {topic}",
                consumer_problem="",
                evidence=message,
                interpretation="",
                topic=topic,
                error=message,
            )

        return parse_insight_response(_extract_text(raw_response), topic=topic)

    def generate_insights(self, topics: List[str], k: Optional[int] = None) -> List[ConsumerInsight]:
        return [self.generate_insight(topic, k=k) for topic in topics]


DEFAULT_INSIGHT_TOPICS = [
    "Digital wallet adoption factors",
    "Barriers to mobile payment adoption",
    "Trust and security concerns in digital payments",
    "Generational differences in payment behavior",
]
