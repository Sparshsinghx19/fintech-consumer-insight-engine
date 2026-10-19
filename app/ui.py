"""
Presentation helpers for the Streamlit interface.

Small, reusable rendering functions plus the app's scoped CSS. No
business logic lives here: these functions only display data produced by
the ingestion, retrieval, RAG, insight, and reporting modules.
"""
from __future__ import annotations

import html
import re
from typing import Iterable, Optional

import streamlit as st

from app.config import Settings
from app.indexing import (
    CHANGED,
    DUPLICATE,
    INDEXED,
    NO_TEXT,
    NOT_INDEXED,
    REMOVED,
    UNREADABLE,
    DocumentStatus,
)
from app.insights import UNSTRUCTURED_PLACEHOLDER, ConsumerInsight
from app.retriever import RetrievedPassage

APP_NAME = "FinTech Consumer Insight Engine"
APP_TAGLINE = "AI-powered FinTech research and consumer insights."

_CSS = """
<style>
:root {
  --fci-navy: #0B1F3A;
  --fci-navy-2: #132B4D;
  --fci-accent: #0F766E;
  --fci-accent-soft: #E3F2EF;
  --fci-surface: #FFFFFF;
  --fci-border: #DFE4EB;
  --fci-text: #0F172A;
  --fci-muted: #5B6475;
}

/* Layout */
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1 { font-size: 1.65rem !important; font-weight: 650 !important; letter-spacing: -0.01em; }
h2 { font-size: 1.2rem !important; font-weight: 620 !important; }
h3 { font-size: 1.02rem !important; font-weight: 620 !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--fci-navy); }
section[data-testid="stSidebar"] * { color: #D9E1EC; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
  border-radius: 8px; padding: 0.35rem 0.6rem; margin: 1px 0;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover { background: var(--fci-navy-2); }
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"],
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"].active {
  background: var(--fci-navy-2); box-shadow: inset 3px 0 0 #2DD4BF;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] p { font-weight: 500; }

/* Cards: metrics, bordered containers, chat messages */
div[data-testid="stMetric"] {
  background: var(--fci-surface); border: 1px solid var(--fci-border);
  border-radius: 10px; padding: 0.85rem 1rem; box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
div[data-testid="stMetric"] label p { color: var(--fci-muted); font-size: 0.82rem; }
div[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 620; }
/* White surface only for the bordered container whose own block holds the
   marker (child combinators stop outer wrappers from matching). */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] .fci-card-marker) {
  background: var(--fci-surface); border-radius: 10px; box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
div[data-testid="element-container"]:has(.fci-card-marker),
div[data-testid="element-container"]:has(.fci-list-marker),
div[data-testid="element-container"]:has(.fci-workspace-marker) { display: none; }

/* Research workspace: give the main and side columns minimum widths so the
   side panel wraps below the main column on medium-width screens instead
   of being squeezed (Streamlit only stacks columns below 640px). */
div[data-testid="stHorizontalBlock"]:has(.fci-workspace-marker) > div[data-testid="column"] { min-width: 230px; }
div[data-testid="stHorizontalBlock"]:has(.fci-workspace-marker) > div[data-testid="column"]:first-child { min-width: min(100%, 440px); }

/* List-style buttons (saved queries, recent questions): left-aligned text. */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] .fci-list-marker) button {
  justify-content: flex-start; text-align: left; min-height: 2.4rem;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] .fci-list-marker) button p {
  text-align: left; font-size: 0.86rem; line-height: 1.35;
}
div[data-testid="stChatMessage"] {
  background: var(--fci-surface); border: 1px solid var(--fci-border); border-radius: 12px; padding: 0.9rem 1rem;
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--fci-accent-soft); border-color: #C7E5DF;
}

/* Brand, badges, small text */
.fci-brand { padding: 0.25rem 0 0.9rem 0; }
.fci-brand-name { font-size: 1.05rem; font-weight: 700; color: #FFFFFF !important; line-height: 1.3; }
.fci-brand-tag { font-size: 0.8rem; color: #9FB0C7 !important; margin-top: 0.2rem; }
.fci-side-label { font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: #8EA0B8 !important; margin: 0.4rem 0 0.3rem 0; }
.fci-side-text { font-size: 0.84rem; line-height: 1.45; }
.fci-subtitle { color: var(--fci-muted); margin-top: -0.6rem; margin-bottom: 1.2rem; font-size: 0.98rem; }
.fci-badge {
  display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px; font-size: 0.76rem;
  font-weight: 600; line-height: 1.5; margin-right: 0.35rem; border: 1px solid transparent; white-space: nowrap;
}
.fci-tone-positive { background: #E3F2EF; color: #0B5F58; border-color: #BFE3DC; }
.fci-tone-warning { background: #FEF3C7; color: #8A4B0A; border-color: #F6DE9C; }
.fci-tone-negative { background: #FEE4E2; color: #A11B1B; border-color: #F8C5C1; }
.fci-tone-neutral { background: #EEF1F5; color: #3B4556; border-color: #DCE1E8; }
.fci-tone-info { background: #E6EEFB; color: #1D4C9E; border-color: #C9D8F3; }
.fci-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.fci-section-label { font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; color: var(--fci-muted); font-weight: 600; margin-bottom: 0.25rem; }
.fci-meta { color: var(--fci-muted); font-size: 0.85rem; }
.fci-card-marker, .fci-list-marker, .fci-workspace-marker { display: none; }
</style>
"""

_TONE_DOT = {
    "positive": "#2DD4BF",
    "warning": "#FBBF24",
    "negative": "#F87171",
    "neutral": "#94A3B8",
}

DOCUMENT_STATUS_BADGES = {
    INDEXED: ("Indexed", "positive"),
    CHANGED: ("Changed — reindex", "warning"),
    NOT_INDEXED: ("Not indexed", "warning"),
    DUPLICATE: ("Duplicate content", "neutral"),
    NO_TEXT: ("No extractable text", "negative"),
    UNREADABLE: ("Unreadable", "negative"),
    REMOVED: ("Removed from disk", "warning"),
}


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def card_marker() -> None:
    """Place inside a bordered container to give it the white card surface."""
    st.markdown('<span class="fci-card-marker"></span>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<p class="fci-subtitle">{html.escape(subtitle)}</p>', unsafe_allow_html=True)


def badge_html(text: str, tone: str = "neutral") -> str:
    return f'<span class="fci-badge fci-tone-{tone}">{html.escape(text)}</span>'


def badges(items: Iterable[tuple[str, str]]) -> None:
    st.markdown(" ".join(badge_html(text, tone) for text, tone in items), unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="fci-section-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def format_bytes(size: Optional[int]) -> str:
    if size is None:
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size} B"


# -- sidebar --------------------------------------------------------------


def sidebar_brand() -> None:
    st.markdown(
        f'<div class="fci-brand"><div class="fci-brand-name">{APP_NAME}</div>'
        f'<div class="fci-brand-tag">{APP_TAGLINE}</div></div>',
        unsafe_allow_html=True,
    )


def sidebar_provider(settings: Settings, model: str, reachable: bool, model_available: bool) -> None:
    gemini = settings.llm_provider == "gemini"
    if reachable and model_available:
        tone, state = "positive", "Available"
    elif reachable:
        tone, state = "warning", "Model unavailable"
    else:
        tone, state = "negative", "Unavailable"
    provider = {"ollama": "Ollama (local)", "gemini": "Gemini API (cloud)"}.get(settings.llm_provider, settings.llm_provider)
    st.markdown(
        f'<div class="fci-side-label">Inference</div>'
        f'<div class="fci-side-text"><b>{html.escape(provider)}</b><br>{html.escape(model)}<br>'
        f'<span class="fci-dot" style="background:{_TONE_DOT[tone]}"></span>{state}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="fci-side-label" style="margin-top:0.9rem">Privacy</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="fci-side-text">{html.escape(short_privacy_notice(settings))}</div>', unsafe_allow_html=True)


# -- privacy --------------------------------------------------------------


def short_privacy_notice(settings: Settings) -> str:
    if settings.llm_provider == "gemini":
        return (
            "Cloud inference: questions and the retrieved passages used for each answer are sent to "
            "Google's Gemini API. Embeddings and the index stay in this app's runtime."
        )
    return "Local inference: generation runs on Ollama and embeddings run locally. No content is sent to a cloud LLM."


def privacy_notice(settings: Settings) -> None:
    if settings.llm_provider == "gemini":
        st.warning(
            "**Cloud inference (Gemini).** Each question or topic, together with the retrieved document "
            "passages (file names and page numbers included), is sent to Google's Gemini API to generate "
            "the response. Full PDFs are not uploaded, and embeddings and the vector index remain in this "
            "application's runtime. This mode is not fully local.",
            icon=":material/cloud:",
        )
    else:
        st.success(
            "**Local inference (Ollama).** Generation runs on your Ollama server and embeddings are computed "
            "locally. Questions, documents, and answers are not sent to a cloud LLM.",
            icon=":material/lock:",
        )


DEMO_CORPUS_NOTE = (
    "The bundled documents are illustrative demo research written for this showcase — "
    "not actual consumer survey results."
)


def demo_banner(settings: Settings) -> None:
    """Notice shown above every page in public demo mode."""
    text = "**Public demo.** The document collection is read-only and usage is rate-limited."
    if settings.llm_provider == "gemini":
        text += (
            " Questions and the retrieved document passages used for each answer are sent to Google's "
            "Gemini API to generate responses — please don't enter confidential or personal information."
        )
    st.info(text, icon=":material/public:")


# -- documents ------------------------------------------------------------


def document_badge(doc: DocumentStatus) -> str:
    label, tone = DOCUMENT_STATUS_BADGES.get(doc.status, (doc.status, "neutral"))
    return badge_html(label, tone)


def document_detail(doc: DocumentStatus) -> str:
    if doc.status == DUPLICATE and doc.duplicate_of:
        return f"Same content as {doc.duplicate_of}"
    if doc.chunks:
        return f"{doc.pages} page{'s' if doc.pages != 1 else ''} · {doc.chunks} chunk{'s' if doc.chunks != 1 else ''}"
    if doc.status == NO_TEXT:
        return "Scanned or image-only; OCR is not supported"
    return "—"


# -- evidence -------------------------------------------------------------


SIMILARITY_HELP = (
    "Cosine similarity between the query and the passage embedding (0–1, higher is closer). "
    "Useful for ranking; it is not a probability or a measure of correctness."
)


def evidence_card(
    passage: RetrievedPassage,
    index: int,
    in_context: Optional[bool] = None,
    preview_chars: int = 520,
    key_prefix: str = "ev",
) -> None:
    """A bordered card for one retrieved passage with its source, page,
    and similarity score; long passages can be expanded."""
    with st.container(border=True):
        card_marker()
        head, score = st.columns([4, 1])
        head.markdown(f"**{index}. {passage.source}** <span class='fci-meta'>· page {passage.page}</span>", unsafe_allow_html=True)
        score.markdown(
            f"<div style='text-align:right'>{badge_html(f'Similarity {passage.relevance_score:.2f}', 'info')}</div>",
            unsafe_allow_html=True,
            help=SIMILARITY_HELP,
        )
        if in_context is False:
            st.caption("Retrieved but not sent to the model: the context limit (MAX_CONTEXT_CHARS) was reached.")
        text = passage.text.strip()
        if len(text) <= preview_chars:
            st.markdown(_quote(text))
        else:
            st.markdown(_quote(text[:preview_chars].rsplit(" ", 1)[0] + " …"))
            with st.expander("Show full passage"):
                st.markdown(_quote(text))


def _quote(text: str) -> str:
    """Render passage text as plain prose: escape Markdown so headings,
    lists, or stray symbols in PDF text don't change the layout."""
    escaped = re.sub(r"([\\`*_{}\[\]<>()#+\-.!|~])", r"\\\1", text)
    return escaped.replace("\n", "  \n")


def evidence_by_document(passages: list[RetrievedPassage], in_context: list[bool]) -> list[dict]:
    """Group retrieved passages by document for the evidence explorer,
    using only the retrieved data (pages, counts, similarity scores)."""
    groups: dict[str, dict] = {}
    for passage, used in zip(passages, in_context):
        group = groups.setdefault(passage.source, {"pages": set(), "passages": 0, "best": 0.0, "sent": 0})
        group["pages"].add(passage.page)
        group["passages"] += 1
        group["best"] = max(group["best"], passage.relevance_score)
        group["sent"] += int(used)
    rows = [
        {
            "Document": source,
            "Pages": ", ".join(str(p) for p in sorted(g["pages"])),
            "Passages": g["passages"],
            "Sent to model": g["sent"],
            "Best similarity": round(g["best"], 4),
        }
        for source, g in groups.items()
    ]
    return sorted(rows, key=lambda row: row["Best similarity"], reverse=True)


def evidence_table_markdown(rows: list[dict]) -> str:
    """Render `evidence_by_document` rows as a Markdown table."""
    if not rows:
        return ""
    headers = list(rows[0].keys())

    def cell(value) -> str:
        text = f"{value:.2f}" if isinstance(value, float) else str(value)
        return text.replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join(lines)


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def workspace_marker() -> None:
    """Place in a column of a two-column workspace layout (see CSS)."""
    st.markdown('<span class="fci-workspace-marker"></span>', unsafe_allow_html=True)


def list_marker() -> None:
    """Place inside a card whose buttons are list items (left-aligned text)."""
    st.markdown('<span class="fci-list-marker"></span>', unsafe_allow_html=True)


def source_chips(passages: list[RetrievedPassage]) -> None:
    seen: list[str] = []
    for p in passages:
        label = f"{p.source} · p. {p.page}"
        if label not in seen:
            seen.append(label)
    badges((label, "neutral") for label in seen)


_SOURCES_HEADING_RE = re.compile(r"^\s*\**\s*SOURCES USED\s*:?\s*\**\s*$", re.IGNORECASE | re.MULTILINE)
_ANSWER_LABEL_RE = re.compile(r"^\s*\**\s*ANSWER\s*:\s*\**\s*", re.IGNORECASE)


def split_model_answer(text: str) -> tuple[str, str]:
    """Split a RAG answer in the prompt's ANSWER / SOURCES USED format into
    (answer body, model-cited sources). Text without that structure is
    returned unchanged as the body."""
    match = _SOURCES_HEADING_RE.search(text)
    body, cited = (text[: match.start()], text[match.end():]) if match else (text, "")
    body = _ANSWER_LABEL_RE.sub("", body.strip(), count=1).strip()
    return body or text.strip(), cited.strip()


# -- insights -------------------------------------------------------------

_CONFIDENCE_TONES = {"high": "positive", "medium": "info", "low": "warning"}


def render_insight(insight: ConsumerInsight) -> None:
    with st.container(border=True):
        card_marker()
        if insight.error:
            st.markdown(f"### {insight.title}")
            st.error(insight.error)
            return

        st.markdown(f"### {insight.title}")
        confidence = (insight.confidence or "").strip()
        tone = _CONFIDENCE_TONES.get(confidence.split()[0].lower() if confidence else "", "neutral")
        items = [(f"Confidence: {confidence}", tone)] if confidence else []
        if insight.topic:
            items.append((f"Topic: {insight.topic}", "neutral"))
        badges(items)

        unstructured = insight.consumer_problem == UNSTRUCTURED_PLACEHOLDER
        if unstructured:
            st.warning("The model did not follow the structured format, so its full response is shown as evidence.")
        elif insight.consumer_problem:
            section_label("Consumer problem")
            st.markdown(insight.consumer_problem)

        left, right = st.columns(2, gap="medium")
        with left:
            section_label("Evidence")
            badges([("From retrieved sources", "positive")])
            st.markdown(insight.evidence or "_No evidence returned._")
            if insight.sources:
                st.caption(f"Sources cited by the model: {insight.sources}")
        with right:
            if insight.interpretation:
                section_label("Interpretation")
                badges([("Inference — not stated in sources", "info")])
                st.markdown(insight.interpretation)

        recommendations = [
            ("Business implication", insight.business_implication),
            ("Recommended action", insight.recommended_action),
            ("Potential opportunity", insight.potential_opportunity),
        ]
        present = [(label, text) for label, text in recommendations if text]
        if present:
            st.divider()
            badges([("Recommendations — not research findings", "warning")])
            cols = st.columns(len(present), gap="medium")
            for col, (label, text) in zip(cols, present):
                with col:
                    section_label(label)
                    st.markdown(text)


# -- reports --------------------------------------------------------------


def report_preview_markdown(markdown: str) -> str:
    """Demote every heading one level for in-page display (the report uses
    H1 for each section). The downloaded file is left unchanged."""
    return re.sub(r"(?m)^(#{1,5}) ", r"#\1 ", markdown)


def empty_state(title: str, body: str, icon: str = ":material/info:") -> None:
    st.info(f"**{title}**  \n{body}", icon=icon)
