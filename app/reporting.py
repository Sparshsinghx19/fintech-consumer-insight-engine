"""
"I didn't want to retrieve a single set of passages and use that same context to generate the entire report, because different sections require different evidence. So I designed the report generator to work section by section. For each section, such as adoption barriers, trust and security, or consumer segments, I run a section-specific retrieval query, pass the most relevant evidence to Mistral, and generate that section.

I then combine the sections into a Markdown report. I also include a Key Evidence section containing directly retrieved passages with their source and page numbers, so the underlying evidence remains traceable. Finally, I clearly separate research evidence from strategic recommendations rather than presenting recommendations as research findings."
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional, Protocol

from app.config import Settings, settings as default_settings
from app.prompts import build_report_group_prompt, build_report_limitations_prompt, build_report_section_prompt
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


@dataclass(frozen=True)
class ReportSectionSpec:
    title: str
    guidance: str
    # Retrieval query ("{topic}" is substituted); None = no retrieval.
    query: Optional[str]


_SECTION_QUERIES = dict(REPORT_SECTIONS)

# Compact reports write the same 10 sections in 3 grouped model calls. Each
# group retrieves evidence with its sections' own queries, so every section
# still sees section-specific passages.
REPORT_GROUPS: list[tuple[str, str, list[ReportSectionSpec]]] = [
    (
        "overview",
        "Overview, trends, and pain points",
        [
            ReportSectionSpec("Executive Summary", "a concise overview of the most important evidence-backed points", "{topic}"),
            ReportSectionSpec("Key Consumer Trends", "notable consumer trends shown in the evidence", _SECTION_QUERIES["Key Consumer Trends"]),
            ReportSectionSpec("Consumer Pain Points", "problems and frustrations consumers experience", _SECTION_QUERIES["Consumer Pain Points"]),
        ],
    ),
    (
        "adoption",
        "Adoption, trust, and consumer segments",
        [
            ReportSectionSpec("Adoption Drivers", "what drives or encourages adoption", _SECTION_QUERIES["Adoption Drivers"]),
            ReportSectionSpec("Adoption Barriers", "what slows or prevents adoption", _SECTION_QUERIES["Adoption Barriers"]),
            ReportSectionSpec("Trust and Security Factors", "trust, security, and privacy factors", _SECTION_QUERIES["Trust and Security Factors"]),
            ReportSectionSpec("Consumer Segments", "differences between consumer groups", _SECTION_QUERIES["Consumer Segments"]),
        ],
    ),
    (
        "strategy",
        "Opportunities, recommendations, and limitations",
        [
            ReportSectionSpec("Actionable Opportunities", "business opportunities suggested by the evidence", "business opportunities related to {topic}"),
            ReportSectionSpec(
                "Strategic Recommendations",
                "recommendations for FinTech companies, clearly framed as recommendations derived from the "
                "evidence, never as research findings",
                "strategic recommendations for FinTech companies regarding {topic}",
            ),
            ReportSectionSpec(
                "Research Limitations",
                "limitations of this analysis based only on the corpus notes (coverage gaps, top-k retrieval "
                "rather than the full corpus, findings reflect only the indexed documents); do not invent others",
                None,
            ),
        ],
    ),
]
REPORT_COMPACT_STEPS = len(REPORT_GROUPS)


@dataclass
class ReportGroupResult:
    """One grouped model call in a compact report."""

    key: str
    title: str
    sections: dict[str, str]  # generated section bodies by title
    passages: list  # passages actually sent to the model (deduplicated)
    error: Optional[str] = None  # set when the model call failed
    missing: List[str] = field(default_factory=list)  # sections absent from the response

    @property
    def ok(self) -> bool:
        return self.error is None and not self.missing


@dataclass
class ReportResult:
    topic: str
    generated_at: str
    markdown: str
    sources: List[str] = field(default_factory=list)
    # One user-facing message per section (or group) that could not be generated.
    errors: List[str] = field(default_factory=list)
    mode: str = "detailed"  # "detailed" (one call per section) or "compact" (grouped)
    # Compact reports only: per-group results, kept so failed groups can be
    # retried without regenerating the successful ones.
    groups: List[ReportGroupResult] = field(default_factory=list)
    key_passages: list = field(default_factory=list)
    k: Optional[int] = None

    @property
    def failed_groups(self) -> List[ReportGroupResult]:
        return [g for g in self.groups if not g.ok]


# Called as progress_callback(completed_steps, total_steps, label) before
# each LLM call during report generation.
ProgressCallback = Callable[[int, int, str], None]

# Executive summary + topic sections + opportunities + recommendations + limitations.
REPORT_LLM_STEPS = 1 + len(REPORT_SECTIONS) + 2 + 1


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
        self._progress: Optional[ProgressCallback] = None
        self._steps_done = 0
        self._errors: List[str] = []

    def _get_llm(self) -> SupportsInvoke:
        if self._llm is None:
            from app.llm import get_chat_model

            self._llm = get_chat_model(self.settings)
        return self._llm

    def _report_progress(self, label: str) -> None:
        if self._progress is not None:
            self._progress(self._steps_done, REPORT_LLM_STEPS, label)
        self._steps_done += 1

    def _generate_section(self, section_title: str, query: str, topic: str, k: Optional[int]) -> tuple[str, list]:
        self._report_progress(section_title)
        passages = self.retriever.search(query, k=k)
        context = format_context(passages, self.settings.max_context_chars)
        prompt = build_report_section_prompt(section_title=section_title, topic=topic, context=context)
        try:
            response = self._get_llm().invoke(prompt)
            text = _extract_text(response).strip()
        except Exception as exc:  # noqa: BLE001
            from app.llm import describe_llm_error

            logger.error("Report section '%s' failed to generate: %s", section_title, exc)
            message = describe_llm_error(exc, self.settings)
            self._errors.append(f"{section_title}: {message}")
            text = f"_This section could not be generated. {message}_"
        return text, passages

    def generate_report(
        self, topic: str, k: Optional[int] = None, progress_callback: Optional[ProgressCallback] = None
    ) -> ReportResult:
        self._progress = progress_callback
        self._steps_done = 0
        self._errors = []
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
        self._report_progress("Research Limitations")
        try:
            limitations_text = _extract_text(self._get_llm().invoke(limitations_prompt)).strip()
        except Exception as exc:  # noqa: BLE001
            from app.llm import describe_llm_error

            logger.error("Limitations section failed: %s", exc)
            self._errors.append(f"Research Limitations: {describe_llm_error(exc, self.settings)}")
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

        if self._progress is not None:
            self._progress(REPORT_LLM_STEPS, REPORT_LLM_STEPS, "Done")
        markdown = "\n".join(lines)
        return ReportResult(
            topic=topic,
            generated_at=generated_at,
            markdown=markdown,
            sources=sorted(all_sources.keys()),
            errors=list(self._errors),
        )

    # -- compact (grouped) reports --------------------------------------------

    def generate_compact_report(
        self, topic: str, k: Optional[int] = None, progress_callback: Optional[ProgressCallback] = None
    ) -> ReportResult:
        """Write all report sections in REPORT_COMPACT_STEPS grouped model
        calls. A failed group doesn't stop the others; it is recorded so it
        can be retried with `retry_failed_groups`."""
        groups: List[ReportGroupResult] = []
        for index, (key, title, _) in enumerate(REPORT_GROUPS):
            if progress_callback is not None:
                progress_callback(index, REPORT_COMPACT_STEPS, title)
            groups.append(self._generate_group(key, topic, k, _sources_of(groups)))
        key_passages = self.retriever.search(topic, k=k)
        if progress_callback is not None:
            progress_callback(REPORT_COMPACT_STEPS, REPORT_COMPACT_STEPS, "Done")
        return self._assemble_compact(topic, groups, key_passages, k)

    def retry_failed_groups(
        self, report: ReportResult, progress_callback: Optional[ProgressCallback] = None
    ) -> ReportResult:
        """Regenerate only the failed or incomplete groups of a compact
        report; successful groups are reused without any model call. If a
        retry fails again, previously generated sections are kept."""
        groups = list(report.groups)
        failed = [i for i, g in enumerate(groups) if not g.ok]
        for step, index in enumerate(failed):
            old = groups[index]
            if progress_callback is not None:
                progress_callback(step, len(failed), old.title)
            others = groups[:index] + groups[index + 1 :]
            new = self._generate_group(old.key, report.topic, report.k, _sources_of(others))
            sections = {**old.sections, **new.sections}
            titles = [spec.title for spec in _group_specs(old.key)]
            groups[index] = ReportGroupResult(
                key=old.key,
                title=old.title,
                sections=sections,
                passages=_dedupe_passages(old.passages + new.passages),
                error=new.error,
                missing=[t for t in titles if not sections.get(t)] if new.error is None else old.missing,
            )
        if progress_callback is not None:
            progress_callback(len(failed), len(failed), "Done")
        return self._assemble_compact(report.topic, groups, report.key_passages, report.k)

    def _generate_group(
        self, key: str, topic: str, k: Optional[int], prior_sources: set[str]
    ) -> ReportGroupResult:
        title = next(t for g, t, _ in REPORT_GROUPS if g == key)
        specs = _group_specs(key)
        ranked = [self.retriever.search(spec.query.format(topic=topic), k=k) for spec in specs if spec.query]
        passages = _interleave_unique(ranked)
        context = format_context(passages, self.settings.max_context_chars)
        # Only passages that survived the context-length cap were seen by the model.
        used = [p for i, p in enumerate(passages, start=1) if f"[Passage {i}] " in context]

        extra = ""
        if any(spec.query is None for spec in specs):
            documents = sorted(prior_sources | {p.source for p in used})
            extra = (
                "CORPUS NOTES (use only for Research Limitations): passages for this report came from: "
                f"{', '.join(documents) or 'no documents'}. Retrieval used the top-{k or self.settings.retrieval_k} "
                "most similar chunks per section query, not the full corpus."
            )
        prompt = build_report_group_prompt(topic, context, [(s.title, s.guidance) for s in specs], extra)
        try:
            text = _extract_text(self._get_llm().invoke(prompt))
        except Exception as exc:  # noqa: BLE001
            from app.llm import describe_llm_error

            logger.error("Report group '%s' failed to generate: %s", title, exc)
            return ReportGroupResult(key, title, {}, used, error=describe_llm_error(exc, self.settings))

        sections = parse_grouped_sections(text, [s.title for s in specs])
        missing = [s.title for s in specs if not sections.get(s.title)]
        return ReportGroupResult(key, title, sections, used, missing=missing)

    def _assemble_compact(
        self, topic: str, groups: List[ReportGroupResult], key_passages: list, k: Optional[int]
    ) -> ReportResult:
        from app.llm import provider_label

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        sections: dict[str, str] = {}
        failed_titles: List[str] = []
        errors: List[str] = []
        for group in groups:
            for spec in _group_specs(group.key):
                body = group.sections.get(spec.title)
                if body:
                    sections[spec.title] = body
                    continue
                failed_titles.append(spec.title)
                if group.error:
                    sections[spec.title] = f"_Not generated: {group.error}_"
                else:
                    sections[spec.title] = "_Not generated: the model's response did not include this section._"
            if group.error:
                errors.append(f"{group.title}: {group.error}")
            for title in group.missing if not group.error else []:
                errors.append(f"{title}: the model's response did not include this section.")

        failed_sections = len(failed_titles)
        lines = [
            f"# FinTech Consumer Insight Report: {topic}",
            "",
            f"*Generated {generated_at} — {provider_label(self.settings)}, evidence-grounded RAG pipeline "
            f"(compact: {REPORT_COMPACT_STEPS} grouped model calls).*",
            "",
        ]
        if failed_sections:
            lines += [
                f"> **Incomplete report:** {failed_sections} of {len(sections)} sections could not be generated. "
                "They are marked below; retry them in the Report Builder.",
                "",
            ]

        def add(title: str, preface: Optional[str] = None) -> None:
            lines.append(f"# {title}")
            if preface:
                lines.append(preface)
            lines.append(sections[title])
            lines.append("")

        add("Executive Summary")
        lines += [
            "# Research Questions",
            f"- What does the indexed research corpus reveal about **{topic}**?",
            "- What evidence exists for adoption drivers and barriers?",
            "- What role do trust and security play?",
            "- How do consumer segments differ?",
            "",
        ]
        for title, _ in REPORT_SECTIONS:
            add(title)

        lines.append("# Key Evidence")
        if key_passages:
            for p in key_passages:
                lines.append(f"- ({p.source}, p. {p.page}, relevance {p.relevance_score:.2f}): {p.text[:280].strip()}...")
        else:
            lines.append("- No directly relevant passages were retrieved for this topic.")
        lines.append("")

        add("Actionable Opportunities")
        add(
            "Strategic Recommendations",
            "_The following are recommendations derived by interpreting the evidence above. "
            "They are business suggestions, not research findings._",
        )
        add("Research Limitations")

        all_sources: dict[str, set] = {}
        for p in [p for g in groups for p in g.passages] + list(key_passages):
            all_sources.setdefault(p.source, set()).add(p.page)
        lines.append("# Source References")
        if all_sources:
            for source in sorted(all_sources):
                pages = ", ".join(str(p) for p in sorted(all_sources[source]))
                lines.append(f"- **{source}** — pages referenced: {pages}")
        else:
            lines.append("- No sources were retrieved for this topic.")
        lines.append("")

        return ReportResult(
            topic=topic,
            generated_at=generated_at,
            markdown="\n".join(lines),
            sources=sorted(all_sources),
            errors=errors,
            mode="compact",
            groups=groups,
            key_passages=list(key_passages),
            k=k,
        )


_MARKER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?\**\s*SECTION\s*:\s*(?P<title>.+?)\s*\**\s*$", re.IGNORECASE | re.MULTILINE
)


def parse_grouped_sections(text: str, titles: List[str]) -> dict[str, str]:
    """Split a grouped response into {section title: body} using the
    "### SECTION: <title>" markers. Unknown titles are ignored; titles
    are matched case-insensitively."""
    by_key = {t.casefold(): t for t in titles}
    matches = list(_MARKER_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        title = by_key.get(match.group("title").strip().rstrip(":").casefold())
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if title and body:
            sections[title] = body
    return sections


def _group_specs(key: str) -> list[ReportSectionSpec]:
    return next(specs for g, _, specs in REPORT_GROUPS if g == key)


def _passage_key(p) -> tuple:
    return (p.chunk_hash,) if getattr(p, "chunk_hash", "") else (p.source, p.page, p.text)


def _dedupe_passages(passages: list) -> list:
    seen, unique = set(), []
    for p in passages:
        if _passage_key(p) not in seen:
            seen.add(_passage_key(p))
            unique.append(p)
    return unique


def _interleave_unique(ranked: List[list]) -> list:
    """Merge per-section result lists rank by rank (each section's best
    passage first), dropping duplicates, so the context-length cap trims the
    least relevant passages of every section rather than whole sections."""
    merged = []
    for rank in range(max((len(r) for r in ranked), default=0)):
        merged.extend(r[rank] for r in ranked if rank < len(r))
    return _dedupe_passages(merged)


def _sources_of(groups: List[ReportGroupResult]) -> set[str]:
    return {p.source for g in groups for p in g.passages}
