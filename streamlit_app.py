"""
FinTech Consumer Insight Engine - Streamlit UI.

Local-first RAG application: Ollama + Mistral + local embeddings +
ChromaDB. No document content, question, or generated content is ever
sent to a cloud LLM API.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from app.chunking import DocumentChunker
from app.config import settings
from app.ingestion import PDFIngestor
from app.insights import DEFAULT_INSIGHT_TOPICS, InsightEngine
from app.llm import check_ollama_status
from app.rag_chain import RAGChain
from app.reporting import ReportGenerator
from app.retriever import SemanticRetriever
from app.vector_store import VectorStoreManager

st.set_page_config(page_title="FinTech Consumer Insight Engine", layout="wide")
settings.ensure_directories()


@st.cache_resource(show_spinner=False)
def get_vector_store_manager() -> VectorStoreManager:
    return VectorStoreManager(settings)


def get_retriever() -> SemanticRetriever:
    return SemanticRetriever(vector_store_manager=get_vector_store_manager(), settings=settings)


def build_index(rebuild: bool = False) -> dict:
    """Run ingestion + chunking + indexing and return a stats dict."""
    ingestor = PDFIngestor(settings)
    pages, report = ingestor.ingest_directory()

    chunker = DocumentChunker(settings)
    documents = chunker.chunk_pages(pages)

    vsm = get_vector_store_manager()
    if rebuild:
        added = vsm.rebuild(documents)
    else:
        added = vsm.add_documents(documents, skip_duplicates=True)

    return {
        "processed_files": report.processed_files,
        "skipped_duplicates": report.skipped_duplicates,
        "failed_files": report.failed_files,
        "empty_or_scanned_files": report.empty_or_scanned_files,
        "total_pages": report.total_pages,
        "total_chunks_this_run": len(documents),
        "chunks_added": added,
    }


st.title("💳 FinTech Consumer Insight Engine")
st.caption("Local-first RAG: Ollama + Mistral + local embeddings + ChromaDB. Nothing leaves your machine.")

tabs = st.tabs(
    ["Dashboard", "Document Ingestion", "Semantic Search", "Ask the Research Assistant", "Generate Insights", "Generate Report"]
)

# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------
with tabs[0]:
    st.subheader("System Status")
    vsm = get_vector_store_manager()
    chunk_count = vsm.count()
    sources = vsm.indexed_sources()
    ollama_status = check_ollama_status(settings)

    col1, col2, col3 = st.columns(3)
    col1.metric("Indexed documents", len(sources))
    col2.metric("Indexed chunks", chunk_count)
    col3.metric("Vector DB", "Ready" if chunk_count > 0 else "Empty")

    col4, col5, col6 = st.columns(3)
    col4.metric("Ollama connection", "Connected" if ollama_status.reachable else "Unreachable")
    col5.metric("Active model", settings.ollama_model)
    col6.metric("Embedding model", settings.embedding_model.split("/")[-1])

    if not ollama_status.reachable:
        st.error(ollama_status.error)
    elif not ollama_status.model_available:
        st.warning(ollama_status.error)
    else:
        st.success(f"Ollama is running and '{settings.ollama_model}' is available.")

    if sources:
        st.subheader("Indexed documents")
        st.write(sorted(sources))
    else:
        st.info("No documents indexed yet. Go to the 'Document Ingestion' tab to add PDFs.")

# ----------------------------------------------------------------------
# Document Ingestion
# ----------------------------------------------------------------------
with tabs[1]:
    st.subheader("Add research PDFs")
    st.write(f"PDFs are read from `{settings.data_dir}`. You can also upload files below; they will be saved there.")

    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        for uf in uploaded_files:
            dest = settings.data_dir / uf.name
            with open(dest, "wb") as f:
                shutil.copyfileobj(uf, f)
        st.success(f"Saved {len(uploaded_files)} file(s) to {settings.data_dir}")

    existing_pdfs = PDFIngestor(settings).discover_pdfs()
    st.write(f"**{len(existing_pdfs)} PDF file(s)** currently in `{settings.data_dir}`:")
    st.write([p.name for p in existing_pdfs])

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Build / update index (incremental)", type="primary"):
            with st.spinner("Ingesting, chunking, and embedding new documents locally..."):
                stats = build_index(rebuild=False)
            st.success("Index updated.")
            st.json(stats)
    with col_b:
        if st.button("Rebuild index from scratch"):
            with st.spinner("Rebuilding the entire vector store locally..."):
                stats = build_index(rebuild=True)
            st.success("Index rebuilt.")
            st.json(stats)

# ----------------------------------------------------------------------
# Semantic Search
# ----------------------------------------------------------------------
with tabs[2]:
    st.subheader("Semantic Search")
    query = st.text_input("Research question", placeholder="What factors influence digital wallet adoption?")
    top_k = st.slider("Number of passages to retrieve", 1, 20, settings.retrieval_k)
    if st.button("Search") and query:
        retriever = get_retriever()
        with st.spinner("Searching the local vector database..."):
            results = retriever.search(query, k=top_k)
        if not results:
            st.warning("No relevant passages found. Have you built the index yet?")
        for r in results:
            with st.container(border=True):
                st.markdown(f"**{r.source}** — page {r.page} · relevance {r.relevance_score:.2f}")
                st.write(r.text)

# ----------------------------------------------------------------------
# Ask the Research Assistant
# ----------------------------------------------------------------------
with tabs[3]:
    st.subheader("Ask the Research Assistant")
    st.caption("Answers are generated locally by Mistral via Ollama, grounded in retrieved passages only.")
    question = st.text_area("Your question", placeholder="How does trust affect consumer willingness to use digital payments?")
    ask_k = st.slider("Passages to retrieve for this question", 1, 20, settings.retrieval_k, key="ask_k")
    if st.button("Ask") and question:
        chain = RAGChain(retriever=get_retriever(), settings=settings)
        with st.spinner("Retrieving evidence and generating a grounded answer with local Mistral..."):
            result = chain.answer(question, k=ask_k)
        st.markdown("### Answer")
        st.write(result.answer)
        st.markdown("### Retrieved Sources")
        for s in result.sources:
            with st.container(border=True):
                st.markdown(f"**{s.source}** — page {s.page} · relevance {s.relevance_score:.2f}")
                st.write(s.text[:600] + ("..." if len(s.text) > 600 else ""))

# ----------------------------------------------------------------------
# Generate Insights
# ----------------------------------------------------------------------
with tabs[4]:
    st.subheader("Generate Consumer Insights")
    st.caption("Each insight separates EVIDENCE, INTERPRETATION, and RECOMMENDATION.")
    topic_choice = st.selectbox("Choose a topic", DEFAULT_INSIGHT_TOPICS + ["Custom topic..."])
    custom_topic = ""
    if topic_choice == "Custom topic...":
        custom_topic = st.text_input("Enter a custom topic")
    insight_k = st.slider("Passages to retrieve", 1, 20, settings.retrieval_k, key="insight_k")

    if st.button("Generate Insight"):
        topic = custom_topic.strip() if topic_choice == "Custom topic..." else topic_choice
        if not topic:
            st.warning("Please enter a topic.")
        else:
            engine = InsightEngine(retriever=get_retriever(), settings=settings)
            with st.spinner("Generating a structured, evidence-grounded insight with local Mistral..."):
                insight = engine.generate_insight(topic, k=insight_k)
            st.markdown(f"## {insight.title}")
            st.markdown(f"**Confidence:** {insight.confidence}")
            st.markdown(f"**Consumer Problem / Behavior**\n\n{insight.consumer_problem}")
            st.markdown(f"**Evidence**\n\n{insight.evidence}")
            st.markdown(f"**Interpretation**\n\n{insight.interpretation}")
            st.markdown(f"**Sources**\n\n{insight.sources}")
            st.markdown(f"**Business Implication**\n\n{insight.business_implication}")
            st.markdown(f"**Recommended Action**\n\n{insight.recommended_action}")
            st.markdown(f"**Potential Opportunity**\n\n{insight.potential_opportunity}")

# ----------------------------------------------------------------------
# Generate Report
# ----------------------------------------------------------------------
with tabs[5]:
    st.subheader("Generate a Full Research Report")
    report_topic = st.text_input("Research topic / question for the report", placeholder="Digital wallet adoption among younger consumers")
    report_k = st.slider("Passages retrieved per section", 1, 20, settings.retrieval_k, key="report_k")

    if st.button("Generate Report", type="primary") and report_topic:
        generator = ReportGenerator(retriever=get_retriever(), settings=settings)
        with st.spinner("Generating full report locally (this can take a few minutes across sections)..."):
            result = generator.generate_report(report_topic, k=report_k)
        st.markdown(result.markdown)
        st.download_button(
            "Download report as Markdown",
            data=result.markdown,
            file_name=f"fintech_report_{report_topic.replace(' ', '_')[:40]}.md",
            mime="text/markdown",
        )
