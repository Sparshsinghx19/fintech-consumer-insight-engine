"""
FinTech Consumer Insight Engine - Streamlit UI.

RAG application with local embeddings + ChromaDB. Generation uses the
provider selected by LLM_PROVIDER: local Ollama (default - nothing is sent
to a cloud LLM API) or the hosted Gemini API (questions and retrieved
passages are sent to Google).

This file is presentation only: it wires the ingestion, indexing,
retrieval, RAG, insight, and reporting modules into six pages. Rendering
helpers live in app/ui.py.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Optional

import streamlit as st

from app import ui
from app.config import settings
from app.indexing import DocumentStatus, REMOVED, build_index as run_build_index, document_statuses
from app.ingestion import PDFIngestor, save_uploaded_pdf
from app.insights import DEFAULT_INSIGHT_TOPICS, InsightEngine
from app.llm import LLMStatus, active_model_name, check_llm_status
from app.rag_chain import RAGChain
from app.reporting import REPORT_LLM_STEPS, ReportGenerator
from app.research_session import (
    ResearchEntry,
    ResearchHistory,
    SavedQueries,
    answer_markdown,
    entry_from_result,
)
from app.retriever import SemanticRetriever
from app.utils import sha256_file
from app.vector_store import IndexingError, SourceStats, VectorStoreManager

st.set_page_config(page_title=ui.APP_NAME, page_icon=":material/insights:", layout="wide")
settings.ensure_directories()
ui.inject_css()

USING_GEMINI = settings.llm_provider == "gemini"
PROVIDER_NAME = "Gemini" if USING_GEMINI else f"local {settings.ollama_model}"
RESEARCH_EXAMPLES = [
    "What factors drive digital wallet adoption?",
    "How does consumer trust affect willingness to use digital payments?",
    "What role do transaction fees play in payment method choice?",
    "What are the main barriers to mobile payment adoption?",
    "How are consumers shifting from cash and cards to digital payments?",
    "How do younger and older consumers differ in payment behavior?",
]


# ----------------------------------------------------------------------
# Services and cached state
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vector_store_manager() -> VectorStoreManager:
    return VectorStoreManager(settings)


def get_retriever() -> SemanticRetriever:
    return SemanticRetriever(vector_store_manager=get_vector_store_manager(), settings=settings)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_llm_status(provider: str, model: str) -> dict:
    # Shared, non-user-specific status; cached so navigation doesn't ping the
    # provider (or call the Gemini API) on every rerun. Keyed by provider and
    # model only - the API key is read from settings, never passed in.
    # Cached as a plain dict (see _cached_source_stats).
    return dataclasses.asdict(check_llm_status(settings))


def get_llm_status() -> LLMStatus:
    return LLMStatus(**_cached_llm_status(settings.llm_provider, active_model_name(settings)))


@st.cache_data(ttl=60, show_spinner=False)
def _cached_source_stats() -> dict[str, tuple[int, int, str]]:
    # One metadata scan of the collection; cleared after every index change.
    # Cached as plain tuples: st.cache_data pickles values, and app classes
    # can't be pickled reliably across Streamlit's module reloads.
    return {name: (s.chunks, s.pages, s.file_hash) for name, s in get_vector_store_manager().source_stats().items()}


def load_source_stats() -> dict[str, SourceStats]:
    return {name: SourceStats(*values) for name, values in _cached_source_stats().items()}


@st.cache_data(show_spinner=False, max_entries=2000)
def _cached_file_hash(path: str, mtime_ns: int, size: int) -> Optional[str]:
    try:
        return sha256_file(path)
    except OSError:
        return None


def file_hash(path: Path) -> Optional[str]:
    try:
        stat = path.stat()
    except OSError:
        return None
    return _cached_file_hash(str(path), stat.st_mtime_ns, stat.st_size)


def invalidate_index_views() -> None:
    _cached_source_stats.clear()


def current_documents() -> list[DocumentStatus]:
    last_run = st.session_state.get("last_index_run") or {}
    return document_statuses(
        PDFIngestor(settings).discover_pdfs(),
        file_hash,
        load_source_stats(),
        no_text_files=last_run.get("empty_or_scanned_files", []),
    )


def build_index(rebuild: bool = False) -> dict:
    """Run ingestion + chunking + indexing and return a stats dict."""
    return run_build_index(get_vector_store_manager(), settings, rebuild=rebuild)


def flash(kind: str, message: str) -> None:
    """Queue a message to show after the next rerun."""
    st.session_state.setdefault("flash", []).append((kind, message))


def show_flash() -> None:
    for kind, message in st.session_state.pop("flash", []):
        getattr(st, kind)(message)


def save_new_uploads(uploaded_files) -> None:
    """Save each uploaded file once. The uploader keeps returning the same
    files on every rerun, so already-saved uploads (tracked by file_id) are
    skipped - which also stops a deleted document from being written back."""
    saved_ids = st.session_state.setdefault("saved_upload_ids", set())
    saved, rejected = [], []
    for uf in uploaded_files:
        if uf.file_id in saved_ids:
            continue
        saved_ids.add(uf.file_id)
        try:
            dest = save_uploaded_pdf(settings.data_dir, uf.name, uf.getvalue())
        except (ValueError, OSError) as exc:
            rejected.append(f"{uf.name}: {exc}")
        else:
            saved.append(dest.name)
    if saved:
        flash("success", f"Uploaded {len(saved)} file(s): {', '.join(saved)}. Update the index to make them searchable.")
    for message in rejected:
        flash("error", f"Upload rejected — {message}")
    if saved or rejected:
        # A fresh widget key empties the uploader, so saved files aren't
        # shown (or offered) again.
        st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
        st.rerun()


def index_is_empty() -> bool:
    return not load_source_stats()


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------
def page_dashboard() -> None:
    ui.page_header(
        "Dashboard",
        "Upload FinTech and digital-payments research, search it semantically, and generate "
        "evidence-grounded answers, consumer insights, and reports with page-level citations.",
    )
    docs = current_documents()
    on_disk = [d for d in docs if d.status != REMOVED]
    stats = load_source_stats()
    total_chunks = sum(s.chunks for s in stats.values())
    pending = [d for d in docs if d.needs_indexing]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source documents", len(on_disk), help=f"PDF files in {settings.data_dir}")
    c2.metric("Indexed documents", len(stats))
    c3.metric("Indexed chunks", f"{total_chunks:,}")
    if not stats and not on_disk:
        c4.metric("Index status", "Empty")
    elif pending:
        c4.metric("Index status", "Needs update", help=f"{len(pending)} document(s) are new, changed, or removed.")
    else:
        c4.metric("Index status", "Up to date")

    if not stats:
        with st.container(border=True):
            ui.card_marker()
            st.markdown("#### Getting started")
            st.markdown(
                "1. **Add research PDFs** in the Knowledge Base (text-based PDFs; scanned images are not OCR'd).\n"
                "2. **Update the index** to chunk and embed them locally.\n"
                "3. **Search the evidence** or **ask a question** — every answer lists the passages it was based on."
            )
            st.page_link(PAGES["knowledge"], label="Open the Knowledge Base", icon=":material/arrow_forward:")
    elif pending:
        st.warning(
            f"{len(pending)} document(s) are new, changed, or removed since the last indexing run. "
            "Update the index in the Knowledge Base so search results match your files."
        )

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with st.container(border=True):
            ui.card_marker()
            st.markdown("#### Language model")
            status = get_llm_status()
            provider = "Gemini API (cloud)" if USING_GEMINI else "Ollama (local)"
            st.markdown(f"**Provider:** {provider}  \n**Model:** `{active_model_name(settings)}`")
            if not status.reachable:
                st.error(status.error)
            elif not status.model_available:
                st.warning(status.error)
            elif USING_GEMINI:
                ui.badges([("API key accepted · model available", "positive")])
            else:
                ui.badges([("Ollama running · model installed", "positive")])
            ui.privacy_notice(settings)
            if st.button("Recheck status"):
                _cached_llm_status.clear()
                st.rerun()
    with right:
        with st.container(border=True):
            ui.card_marker()
            st.markdown("#### Processing")
            st.markdown(
                f"**Embeddings:** `{settings.embedding_model.split('/')[-1]}` (local)  \n"
                f"**Chunking:** {settings.chunk_size} characters, {settings.chunk_overlap} overlap  \n"
                f"**Retrieval:** top {settings.retrieval_k} passages by default  \n"
                f"**Context limit:** {settings.max_context_chars:,} characters per request  \n"
                f"**Vector store:** ChromaDB, persisted on disk"
            )

    st.markdown("#### Quick actions")
    q1, q2, q3 = st.columns(3, gap="medium")
    actions = [
        (q1, "knowledge", "Upload research", "Add PDFs and update the index.", ":material/upload_file:"),
        (q2, "search", "Search evidence", "Find passages by meaning, with page numbers.", ":material/search:"),
        (q3, "assistant", "Ask a question", "Get an answer grounded in your documents.", ":material/forum:"),
    ]
    for col, page_key, label, description, icon in actions:
        with col, st.container(border=True):
            ui.card_marker()
            st.page_link(PAGES[page_key], label=label, icon=icon)
            st.caption(description)


# ----------------------------------------------------------------------
# Knowledge Base
# ----------------------------------------------------------------------
def page_knowledge_base() -> None:
    ui.page_header("Knowledge Base", "Manage the research PDFs that answers, insights, and reports are built from.")
    show_flash()

    with st.container(border=True):
        ui.card_marker()
        st.markdown("#### Upload PDFs")
        uploaded = st.file_uploader(
            "Upload research PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.get('uploader_key', 0)}",
            help="Text-based PDFs only. Scanned documents need OCR before upload.",
        )
        st.caption(f"Files are saved to `{settings.data_dir}`. Uploading does not index them — update the index below.")
        if uploaded:
            save_new_uploads(uploaded)

    docs = current_documents()
    pending = [d for d in docs if d.needs_indexing]
    with st.container(border=True):
        ui.card_marker()
        st.markdown("#### Documents")
        if not docs:
            ui.empty_state("No documents yet", "Upload PDFs above, or copy them into the data folder.", ":material/folder_open:")
        else:
            on_disk = sum(1 for d in docs if d.status != REMOVED)
            st.caption(f"{on_disk} file(s) on disk · {sum(1 for d in docs if d.chunks and d.status != REMOVED)} indexed · {len(pending)} need indexing")
            header = st.columns([4.4, 2.3, 2.6, 1.1, 1.7])
            for col, title in zip(header, ["File", "Status", "Index details", "Size", ""]):
                col.markdown(f"<span class='fci-meta'>{title}</span>", unsafe_allow_html=True)
            for doc in docs:
                render_document_row(doc)

    with st.container(border=True):
        ui.card_marker()
        st.markdown("#### Indexing")
        inc, full = st.columns(2, gap="large")
        with inc:
            st.markdown("**Update index**")
            st.caption("Embeds new and changed documents and removes chunks of deleted or renamed files. Unchanged documents are skipped.")
            if st.button("Update index", type="primary", disabled=not docs):
                run_indexing(rebuild=False)
        with full:
            st.markdown("**Rebuild from scratch**")
            st.caption("Re-embeds every document. Needed only after changing the embedding model or chunk settings; can take a long time on large collections.")
            confirmed = st.checkbox(
                "I understand this re-embeds every document",
                key=f"confirm_rebuild_{st.session_state.get('rebuild_confirm_key', 0)}",
            )
            if st.button("Rebuild index", disabled=not confirmed):
                run_indexing(rebuild=True)
        render_last_index_run()


def render_document_row(doc: DocumentStatus) -> None:
    name, status, detail, size, action = st.columns([4.4, 2.3, 2.6, 1.1, 1.7], vertical_alignment="center")
    name.markdown(f"**{doc.name}**")
    status.markdown(ui.document_badge(doc), unsafe_allow_html=True)
    detail.markdown(f"<span class='fci-meta'>{ui.document_detail(doc)}</span>", unsafe_allow_html=True)
    size.markdown(f"<span class='fci-meta'>{ui.format_bytes(doc.size_bytes)}</span>", unsafe_allow_html=True)
    with action:
        removed = doc.status == REMOVED
        with st.popover("Remove" if removed else "Delete", use_container_width=True):
            if removed:
                st.markdown(f"Remove the indexed chunks of **{doc.name}**? The file is no longer on disk.")
                confirm_label = "Remove from index"
            else:
                st.markdown(f"Delete **{doc.name}** from disk and remove its chunks from the index? This cannot be undone.")
                confirm_label = "Delete permanently"
            if st.button(confirm_label, key=f"delete_{doc.name}", type="primary"):
                delete_document(doc.name, delete_file=not removed)


def delete_document(name: str, delete_file: bool) -> None:
    try:
        removed_chunks = get_vector_store_manager().delete_by_source(name)
    except Exception as exc:  # noqa: BLE001
        flash("error", f"Could not remove indexed chunks for {name}: {exc}")
        removed_chunks = 0
    if delete_file:
        try:
            (settings.data_dir / name).unlink()
        except OSError as exc:
            flash("error", f"Could not delete {name}: {exc}")
        else:
            flash("success", f"Deleted {name} and removed {removed_chunks} indexed chunk(s).")
    else:
        flash("success", f"Removed {removed_chunks} indexed chunk(s) for {name}.")
    invalidate_index_views()
    st.rerun()


def run_indexing(rebuild: bool) -> None:
    label = "Rebuilding the index" if rebuild else "Updating the index"
    with st.status(f"{label}…", expanded=True) as status:
        st.write("Reading PDFs, chunking text, and computing embeddings locally.")
        try:
            stats = build_index(rebuild=rebuild)
        except IndexingError as exc:
            status.update(label="Indexing failed", state="error")
            st.session_state["last_index_run"] = {"error": str(exc), "rebuild": rebuild}
        else:
            status.update(label="Index rebuilt" if rebuild else "Index updated", state="complete")
            st.session_state["last_index_run"] = {**stats, "rebuild": rebuild}
    # A new key resets the confirmation checkbox (an instantiated widget's
    # value can't be assigned directly).
    st.session_state["rebuild_confirm_key"] = st.session_state.get("rebuild_confirm_key", 0) + 1
    invalidate_index_views()
    st.rerun()


def render_last_index_run() -> None:
    run = st.session_state.get("last_index_run")
    if not run:
        return
    st.divider()
    title = "Last rebuild" if run.get("rebuild") else "Last index update"
    if "error" in run:
        st.error(f"**{title} failed.** {run['error']}")
        return
    st.markdown(f"**{title}** — completed this session")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Files processed", len(run["processed_files"]))
    m2.metric("Chunks added", run["chunks_added"])
    m3.metric("Chunks removed", run.get("chunks_removed", 0))
    m4.metric("Chunks unchanged", run.get("chunks_unchanged", 0))
    notes = [
        ("Skipped as duplicates", run["skipped_duplicates"]),
        ("Could not be read", run["failed_files"]),
        ("No extractable text (scanned)", run["empty_or_scanned_files"]),
    ]
    for label, files in notes:
        if files:
            st.caption(f"{label}: {', '.join(files)}")


# ----------------------------------------------------------------------
# Semantic Search
# ----------------------------------------------------------------------
def page_search() -> None:
    ui.page_header("Semantic Search", "Find the passages most similar in meaning to your query, with source and page.")
    if index_is_empty():
        ui.empty_state("The index is empty", "Add and index documents in the Knowledge Base to search them.")
        st.page_link(PAGES["knowledge"], label="Open the Knowledge Base", icon=":material/arrow_forward:")
        return

    with st.form("search_form", border=True):
        query = st.text_input(
            "Search the knowledge base",
            placeholder="e.g. What role does fee transparency play in payment choice?",
            value=st.session_state.get("search", {}).get("query", ""),
        )
        col_k, col_btn = st.columns([4, 1], vertical_alignment="bottom")
        k = col_k.slider("Passages to retrieve", 1, 20, settings.retrieval_k)
        submitted = col_btn.form_submit_button("Search", type="primary", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Enter a search query.")
        else:
            with st.spinner("Searching…"):
                results = get_retriever().search(query, k=k)
            st.session_state["search"] = {"query": query, "results": results}

    search = st.session_state.get("search")
    if not search:
        return
    results = search["results"]
    if not results:
        st.info(f"No passages matched “{search['query']}”. Try different wording.")
        return
    st.markdown(f"**{len(results)} passages** for “{search['query']}”, most similar first.")
    for i, passage in enumerate(results, start=1):
        ui.evidence_card(passage, i, key_prefix="search")


# ----------------------------------------------------------------------
# Research Assistant
# ----------------------------------------------------------------------
# History and saved queries live only in this browser session's
# st.session_state (never in a module global or st.cache_*), so one user's
# research is never visible to another session.


def research_history() -> ResearchHistory:
    return st.session_state.setdefault("research_history", ResearchHistory())


def saved_queries() -> SavedQueries:
    return st.session_state.setdefault("saved_queries", SavedQueries())


def queue_research(question: str, k: int) -> None:
    st.session_state["pending_research"] = {"question": question, "k": k}


def select_entry(entry_id: Optional[str]) -> None:
    st.session_state["research_selected"] = entry_id


def open_saved_query(question: str) -> None:
    """Show the latest answer for a saved question if it's still in
    history (no model call); otherwise put it in the question box."""
    entry = research_history().latest_for(question)
    if entry is not None:
        select_entry(entry.id)
    else:
        select_entry(None)
        st.session_state["research_draft"] = question


def clear_research_history() -> None:
    research_history().clear()
    select_entry(None)


def page_assistant() -> None:
    ui.page_header(
        "Research Assistant",
        "Ask research questions about your documents and inspect the evidence behind every answer.",
    )
    history = research_history()
    pending = st.session_state.pop("pending_research", None)
    if index_is_empty():
        st.warning("The index is empty, so there is no evidence to answer from. Add documents in the Knowledge Base first.")

    main, side = st.columns([2.7, 1], gap="medium")
    with side:
        ui.workspace_marker()
        render_saved_queries()
        render_recent_questions()
        st.caption(
            "Recent questions and saved queries are kept in memory for this browser session only; "
            "reloading or closing the page clears them."
        )

    with main:
        with st.form("research_form", border=True, clear_on_submit=True):
            question = st.text_area(
                "Research question",
                key="research_draft",
                height=88,
                placeholder="e.g. What role do transaction fees play in payment method choice?",
            )
            k_col, btn_col = st.columns([2.2, 1], vertical_alignment="bottom")
            k = k_col.slider("Passages to retrieve", 1, 20, settings.retrieval_k, key="ask_k")
            submitted = btn_col.form_submit_button("Research", type="primary", use_container_width=True)
            st.caption(
                "Each question is answered independently from passages retrieved for it."
                + (" Cloud inference: the question and retrieved passages are sent to Google's Gemini API." if USING_GEMINI else "")
            )

        if submitted and not question.strip():
            st.warning("Enter a research question.")
        request = pending or ({"question": question.strip(), "k": k} if submitted and question.strip() else None)
        if request:
            run_research(request["question"], request["k"])

        entry = history.get(st.session_state.get("research_selected"))
        if entry is None:
            render_research_examples()
        else:
            render_research_entry(entry)


def run_research(question: str, k: int) -> None:
    with st.spinner(f"Retrieving evidence and generating an answer with {PROVIDER_NAME}…"):
        result = RAGChain(retriever=get_retriever(), settings=settings).answer(question, k=k)
    provider = "Gemini API (cloud)" if USING_GEMINI else "Ollama (local)"
    entry = entry_from_result(result, k=k, provider=provider, model=active_model_name(settings))
    research_history().add(entry)
    select_entry(entry.id)
    st.rerun()


def render_research_examples() -> None:
    with st.container(border=True):
        ui.card_marker()
        st.markdown("#### Start a research question")
        st.caption("Pick an example to run it against your indexed documents, or write your own above.")
        cols = st.columns(2)
        k = st.session_state.get("ask_k", settings.retrieval_k)
        for i, example in enumerate(RESEARCH_EXAMPLES):
            cols[i % 2].button(
                example, key=f"example_{i}", use_container_width=True, on_click=queue_research, args=(example, k)
            )


def render_research_entry(entry: ResearchEntry) -> None:
    saved = saved_queries()
    with st.container(border=True):
        ui.card_marker()
        ui.section_label("Research question")
        st.markdown(f"### {entry.question}")
        meta = [
            (f"{entry.asked_at:%H:%M} UTC", "neutral"),
            (entry.model, "neutral"),
            (f"{len(entry.sources)} passage{'s' if len(entry.sources) != 1 else ''}", "neutral"),
        ]
        if entry.error:
            meta.append(("Generation failed", "negative"))
        ui.badges(meta)

        copy_col, save_col, rerun_col, new_col = st.columns(4)
        with copy_col.popover("Copy", use_container_width=True, disabled=bool(entry.error)):
            st.code(answer_markdown(entry), language="markdown", wrap_lines=True)
            st.caption(
                "Use the copy icon at the top right of the box. If your browser blocks clipboard access, "
                "select the text or download it."
            )
            st.download_button(
                "Download .md",
                data=answer_markdown(entry),
                file_name=f"research_answer_{entry.id}.md",
                mime="text/markdown",
                key=f"dl_{entry.id}",
                use_container_width=True,
            )
        if saved.is_saved(entry.question):
            save_col.button("Unsave", key=f"unsave_{entry.id}", on_click=saved.unsave, args=(entry.question,), use_container_width=True)
        else:
            save_col.button(
                "Save",
                key=f"save_{entry.id}",
                on_click=saved.save,
                args=(entry.question, entry.k),
                disabled=saved.is_full,
                help="Saved-query limit reached; unsave one first." if saved.is_full else None,
                use_container_width=True,
            )
        rerun_col.button(
            "Rerun", key=f"rerun_{entry.id}", on_click=queue_research, args=(entry.question, entry.k),
            help="Ask again with the same settings; the new answer is added to recent questions.",
            use_container_width=True,
        )
        new_col.button("New", key=f"new_{entry.id}", on_click=select_entry, args=(None,), use_container_width=True)

        st.divider()
        if entry.error:
            st.error(entry.error)
        else:
            body, cited = ui.split_model_answer(entry.answer)
            st.markdown(body)
            if cited:
                ui.section_label("Sources cited by the model")
                st.markdown(cited)
                st.caption(
                    "These citations are generated by the model and are not automatically verified — "
                    "compare them with the evidence below."
                )

    render_evidence_explorer(entry)


def render_evidence_explorer(entry: ResearchEntry) -> None:
    with st.container(border=True):
        ui.card_marker()
        st.markdown("#### Evidence explorer")
        if not entry.sources:
            st.info("No passages were retrieved for this question, so the answer has no supporting evidence.")
            return
        used = sum(entry.in_context)
        st.caption(
            f"The {len(entry.sources)} passages retrieved for this question"
            + (f"; {used} fit in the model's context." if used < len(entry.sources) else ", all sent to the model.")
        )
        passages_tab, documents_tab = st.tabs(["Passages", "By document"])
        with passages_tab:
            for n, (passage, in_context) in enumerate(zip(entry.sources, entry.in_context), start=1):
                ui.evidence_card(passage, n, in_context=in_context, key_prefix=f"res{entry.id}")
        with documents_tab:
            # A plain Markdown table: a handful of rows needs no interactive grid.
            st.markdown(ui.evidence_table_markdown(ui.evidence_by_document(entry.sources, entry.in_context)))
            st.caption(ui.SIMILARITY_HELP)


def render_saved_queries() -> None:
    saved = saved_queries()
    with st.container(border=True):
        ui.card_marker()
        ui.list_marker()
        st.markdown(f"#### Saved queries <span class='fci-meta'>{len(saved)}</span>", unsafe_allow_html=True)
        if not len(saved):
            st.caption("Save a question from its answer to keep it here for this session.")
            return
        for i, item in enumerate(saved.items):
            st.button(
                ui.truncate(item.question, 90), key=f"saved_open_{i}", on_click=open_saved_query, args=(item.question,),
                help="Open the latest answer, or load the question if it is no longer in recent history.",
                use_container_width=True,
            )
            run_col, remove_col = st.columns(2)
            run_col.button("Run", key=f"saved_run_{i}", on_click=queue_research, args=(item.question, item.k), use_container_width=True)
            remove_col.button("Unsave", key=f"saved_rm_{i}", on_click=saved.unsave, args=(item.question,), use_container_width=True)


def render_recent_questions() -> None:
    history = research_history()
    selected = st.session_state.get("research_selected")
    with st.container(border=True):
        ui.card_marker()
        ui.list_marker()
        st.markdown(
            f"#### Recent questions <span class='fci-meta'>{len(history)}/{history.max_entries}</span>",
            unsafe_allow_html=True,
        )
        if not len(history):
            st.caption("Questions you ask appear here. Select one to revisit its answer and evidence without re-running it.")
            return
        for entry in history.entries:
            st.button(
                ("⚠ " if entry.error else "") + ui.truncate(entry.question, 90),
                key=f"recent_{entry.id}",
                on_click=select_entry,
                args=(entry.id,),
                type="primary" if entry.id == selected else "secondary",
                help=f"Asked {entry.asked_at:%H:%M} UTC · {entry.model} · {len(entry.sources)} passages",
                use_container_width=True,
            )
        st.button("Clear history", key="clear_history", on_click=clear_research_history, use_container_width=True)


# ----------------------------------------------------------------------
# Insights Studio
# ----------------------------------------------------------------------
def page_insights() -> None:
    ui.page_header(
        "Insights Studio",
        "Generate structured consumer insights that keep source evidence, interpretation, and recommendations separate.",
    )
    insights: list = st.session_state.setdefault("insights", [])

    with st.container(border=True):
        ui.card_marker()
        topic_col, k_col = st.columns([3, 1], vertical_alignment="bottom")
        choice = topic_col.selectbox("Topic", DEFAULT_INSIGHT_TOPICS + ["Custom topic…"])
        k = k_col.slider("Passages", 1, 20, settings.retrieval_k, key="insight_k")
        custom = st.text_input("Custom topic", placeholder="e.g. Buy-now-pay-later usage among students") if choice == "Custom topic…" else ""
        topic = custom.strip() if choice == "Custom topic…" else choice
        gen_col, clear_col, _ = st.columns([1.6, 1.2, 4])
        generate = gen_col.button("Generate insight", type="primary", disabled=not topic, use_container_width=True)
        if clear_col.button("Clear", disabled=not insights, use_container_width=True):
            st.session_state["insights"] = []
            st.rerun()
        if USING_GEMINI:
            st.caption("Cloud inference: the topic and retrieved passages are sent to Google's Gemini API.")

    if generate:
        if index_is_empty():
            st.warning("The index is empty; the insight would have no evidence. Add documents in the Knowledge Base first.")
        else:
            with st.spinner(f"Retrieving evidence and generating an insight with {PROVIDER_NAME}…"):
                insight = InsightEngine(retriever=get_retriever(), settings=settings).generate_insight(topic, k=k)
            insights.insert(0, insight)

    if not insights:
        ui.empty_state("No insights yet", "Choose a topic and generate an insight. Results stay here while you use the app.", ":material/lightbulb:")
        return
    for insight in insights:
        ui.render_insight(insight)


# ----------------------------------------------------------------------
# Report Builder
# ----------------------------------------------------------------------
def page_reports() -> None:
    ui.page_header(
        "Report Builder",
        "Generate a full Markdown research report, written section by section from freshly retrieved evidence.",
    )
    with st.container(border=True):
        ui.card_marker()
        topic = st.text_input("Report topic", placeholder="e.g. Digital wallet adoption among younger consumers")
        k = st.slider("Passages retrieved per section", 1, 20, settings.retrieval_k, key="report_k")
        st.caption(
            f"Runs {REPORT_LLM_STEPS} sequential model calls"
            + (" — with a local model this can take several minutes." if not USING_GEMINI else "; each sends the topic and retrieved passages to Google's Gemini API.")
        )
        generate = st.button("Generate report", type="primary", disabled=not topic.strip())

    if generate:
        if index_is_empty():
            st.warning("The index is empty; the report would have no evidence. Add documents in the Knowledge Base first.")
        else:
            progress = st.progress(0.0, text="Starting…")

            def on_progress(done: int, total: int, label: str) -> None:
                text = "Assembling report…" if done >= total else f"Writing “{label}” ({done + 1} of {total})"
                progress.progress(min(done / total, 1.0), text=text)

            result = ReportGenerator(retriever=get_retriever(), settings=settings).generate_report(
                topic.strip(), k=k, progress_callback=on_progress
            )
            progress.empty()
            st.session_state["report"] = result

    report = st.session_state.get("report")
    if report is None:
        ui.empty_state("No report yet", "Enter a topic and generate a report. It stays here until you generate another or clear it.", ":material/article:")
        return

    with st.container(border=True):
        ui.card_marker()
        head, actions = st.columns([3, 2], vertical_alignment="center")
        head.markdown(f"#### {report.topic}")
        head.caption(f"Generated {report.generated_at} · {len(report.sources)} source document(s)")
        dl, clear = actions.columns(2)
        dl.download_button(
            "Download .md",
            data=report.markdown,
            file_name=f"fintech_report_{re.sub(r'[^A-Za-z0-9_-]+', '_', report.topic)[:40].strip('_') or 'report'}.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )
        if clear.button("Clear", use_container_width=True):
            del st.session_state["report"]
            st.rerun()
        if report.errors:
            st.warning(f"{len(report.errors)} section(s) could not be generated:\n\n" + "\n".join(f"- {e}" for e in report.errors))
        if report.sources:
            ui.badges((source, "neutral") for source in report.sources)

    with st.container(border=True):
        ui.card_marker()
        st.markdown(ui.report_preview_markdown(report.markdown))


# ----------------------------------------------------------------------
# Shell
# ----------------------------------------------------------------------
PAGES = {
    "dashboard": st.Page(page_dashboard, title="Dashboard", icon=":material/space_dashboard:", url_path="dashboard", default=True),
    "knowledge": st.Page(page_knowledge_base, title="Knowledge Base", icon=":material/folder_open:", url_path="knowledge-base"),
    "search": st.Page(page_search, title="Semantic Search", icon=":material/search:", url_path="search"),
    "assistant": st.Page(page_assistant, title="Research Assistant", icon=":material/forum:", url_path="assistant"),
    "insights": st.Page(page_insights, title="Insights Studio", icon=":material/lightbulb:", url_path="insights"),
    "reports": st.Page(page_reports, title="Report Builder", icon=":material/description:", url_path="reports"),
}

navigation = st.navigation(list(PAGES.values()), position="hidden")

with st.sidebar:
    ui.sidebar_brand()
    for page in PAGES.values():
        st.page_link(page)
    st.divider()
    _status = get_llm_status()
    ui.sidebar_provider(settings, active_model_name(settings), _status.reachable, _status.model_available)

navigation.run()
