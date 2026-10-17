"""
"I didn't want to retrieve a single set of passages and use that same context to generate the entire report, because different sections require different evidence. So I designed the report generator to work section by section. For each section, such as adoption barriers, trust and security, or consumer segments, I run a section-specific retrieval query, pass the most relevant evidence to Mistral, and generate that section.

I then combine the sections into a Markdown report. I also include a Key Evidence section containing directly retrieved passages with their source and page numbers, so the underlying evidence remains traceable. Finally, I clearly separate research evidence from strategic recommendations rather than presenting recommendations as research findings."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Protocol

from app.config import Settings, settings as default_settings
from app.prompts import build_report_limitations_prompt, build_report_section_prompt
from app.rag_chain import format_context
from app.retriever import SemanticRetriever
from app.utils import get_logger

logger = get_logger(__name__)


class SupportsInvoke(Protocol):
    def invoke(self, prompt: str): ...


# Maps report section title -> the retrieval query used to gather
# section-specific evidence from the vector store.
REPORT_SECTIONS: list[tuple[str, str]] = [
    ("Key Consumer Trends", "consumer trends in digital payments and financial technology adoption"),
    ("Consumer Pain Points", "consumer pain points and frustrations with digital payments"),
    ("Adoption Drivers", "factors that drive or encourage adoption of digital payments and wallets"),
    ("Adoption Barriers", "barriers and obstacles to adoption of digital payments and mobile wallets"),
    ("Trust and Security Factors", "trust, security, and privacy concerns affecting digital payment adoption"),
    ("Consumer Segments", "differences between consumer segments, demographics, or generations in payment behavior"),
]


@dataclass
class ReportResult:
    topic: str
    generated_at: str
    markdown: str
    sources: List[str] = field(default_factory=list)


def _extract_text(response) -> str:
    if hasattr(response, "content"):
        return response.content
    return str(response)


class ReportGenerator:
    """Generates a complete, evidence-grounded Markdown research report."""

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

    def _generate_section(self, section_title: str, query: str, topic: str, k: Optional[int]) -> tuple[str, list]:
        passages = self.retriever.search(query, k=k)
        context = format_context(passages, self.settings.max_context_chars)
        prompt = build_report_section_prompt(section_title=section_title, topic=topic, context=context)
        try:
            response = self._get_llm().invoke(prompt)
            text = _extract_text(response).strip()
        except Exception as exc:  # noqa: BLE001
            from app.llm import describe_llm_error

            logger.error("Report section '%s' failed to generate: %s", section_title, exc)
            text = f"_This section could not be generated. {describe_llm_error(exc, self.settings)}_"
        return text, passages

    def generate_report(self, topic: str, k: Optional[int] = None) -> ReportResult:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        all_sources: dict[str, set[int]] = {}

        def record_sources(passages):
            for p in passages:
                all_sources.setdefault(p.source, set()).add(p.page)

        lines: List[str] = []
        lines.append(f"# FinTech Consumer Insight Report: {topic}")
        lines.append("")
        from app.llm import provider_label

        lines.append(f"*Generated {generated_at} — {provider_label(self.settings)}, evidence-grounded RAG pipeline.*")
        lines.append("")

        # Executive summary — built from a broad retrieval over the topic itself.
        lines.append("# Executive Summary")
        exec_text, exec_passages = self._generate_section("Executive Summary", topic, topic, k)
        record_sources(exec_passages)
        lines.append(exec_text)
        lines.append("")

        lines.append("# Research Questions")
        lines.append(f"- What does the indexed research corpus reveal about **{topic}**?")
        lines.append("- What evidence exists for adoption drivers and barriers?")
        lines.append("- What role do trust and security play?")
        lines.append("- How do consumer segments differ?")
        lines.append("")

        for section_title, query in REPORT_SECTIONS:
            lines.append(f"# {section_title}")
            text, passages = self._generate_section(section_title, query, topic, k)
            record_sources(passages)
            lines.append(text)
            lines.append("")

        # Key Evidence: a direct, citation-anchored excerpt list (no LLM
        # paraphrasing, so evidence traceability is guaranteed).
        lines.append("# Key Evidence")
        key_passages = self.retriever.search(topic, k=k)
        record_sources(key_passages)
        if key_passages:
            for p in key_passages:
                snippet = p.text[:280].strip()
                lines.append(f"- ({p.source}, p. {p.page}, relevance {p.relevance_score:.2f}): {snippet}...")
        else:
            lines.append("- No directly relevant passages were retrieved for this topic.")
        lines.append("")

        lines.append("# Actionable Opportunities")
        opp_text, opp_passages = self._generate_section(
            "Actionable Opportunities", f"business opportunities related to {topic}", topic, k
        )
        record_sources(opp_passages)
        lines.append(opp_text)
        lines.append("")

        lines.append("# Strategic Recommendations")
        lines.append(
            "_The following are recommendations derived by interpreting the evidence above. "
            "They are business suggestions, not research findings._"
        )
        rec_text, rec_passages = self._generate_section(
            "Strategic Recommendations", f"strategic recommendations for FinTech companies regarding {topic}", topic, k
        )
        record_sources(rec_passages)
        lines.append(rec_text)
        lines.append("")

        lines.append("# Research Limitations")
        limitations_context = (
            f"The corpus indexed at report time contains passages from the following "
            f"documents: {', '.join(sorted(all_sources.keys())) or 'none retrieved'}. "
            f"Retrieval used the top-{k or self.settings.retrieval_k} most similar chunks per "
            f"query rather than the full corpus."
        )
        limitations_prompt = build_report_limitations_prompt(topic=topic, limitations_context=limitations_context)
        try:
            limitations_text = _extract_text(self._get_llm().invoke(limitations_prompt)).strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Limitations section failed: %s", exc)
            limitations_text = (
                "This report is based only on the documents currently indexed in the vector store "
                "and only the top-k retrieved chunks per section, not the entire corpus."
            )
        lines.append(limitations_text)
        lines.append("")

        lines.append("# Source References")
        if all_sources:
            for source in sorted(all_sources.keys()):
                pages = ", ".join(str(p) for p in sorted(all_sources[source]))
                lines.append(f"- **{source}** — pages referenced: {pages}")
        else:
            lines.append("- No sources were retrieved for this topic.")
        lines.append("")

        markdown = "\n".join(lines)
        return ReportResult(topic=topic, generated_at=generated_at, markdown=markdown, sources=sorted(all_sources.keys()))
