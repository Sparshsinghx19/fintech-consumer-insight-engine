# FinTech Consumer Insight Engine

A **local-first Retrieval-Augmented Generation (RAG)** application that ingests large
volumes of unstructured FinTech / digital-payments research (500+ pages of PDFs),
indexes it into a persistent vector database, and uses a **locally running Mistral
model via Ollama** to answer questions, generate structured consumer insights, and
produce full Markdown research reports — all **without sending any document content,
question, or generated output to a cloud LLM API**.

---

## 1. Project Overview

Research teams at FinTech companies often have hundreds of pages of survey reports,
whitepapers, and consumer-behavior studies sitting in PDFs. This project turns that
corpus into a queryable knowledge base:

1. **Ingest** PDFs → extract text + page-level metadata.
2. **Chunk** the text into overlapping, appropriately sized passages.
3. **Embed** each chunk locally (Sentence-Transformers) and store it in **ChromaDB**.
4. **Retrieve** the most relevant chunks for a question via semantic search.
5. **Generate** an evidence-grounded answer, structured insight, or full report using
   **Mistral running locally through Ollama** — never a cloud LLM.

Every generated claim is required (by prompt design) to cite a document name and page
number that actually appears in the retrieved context, and the model is explicitly
instructed never to invent statistics, findings, or citations.

## 2. Architecture

```
                         ┌─────────────────────────┐
                         │   data/raw/*.pdf         │
                         └───────────┬─────────────┘
                                     │  PDFIngestor (pypdf)
                                     ▼
                         ┌─────────────────────────┐
                         │  PageRecord (text +      │
                         │  source/page metadata)   │
                         └───────────┬─────────────┘
                                     │  DocumentChunker
                                     ▼                     (RecursiveCharacterTextSplitter)
                         ┌─────────────────────────┐
                         │  LangChain Document      │
                         │  chunks + metadata       │
                         └───────────┬─────────────┘
                                     │  local HuggingFaceEmbeddings
                                     ▼
                         ┌─────────────────────────┐
                         │  ChromaDB (persisted     │
                         │  in vectorstore/)        │
                         └───────────┬─────────────┘
                                     │  SemanticRetriever.search()
                                     ▼
        User Question ──────▶ Retrieved Passages ──────▶ Grounded Prompt
                                                                 │
                                                                 ▼
                                                   ┌─────────────────────────┐
                                                   │  Mistral via Ollama      │
                                                   │  (local inference only)  │
                                                   └───────────┬─────────────┘
                                                                 │
                                     ┌───────────────────────────┼───────────────────────────┐
                                     ▼                            ▼                            ▼
                            RAGChain.answer()          InsightEngine.generate_insight() ReportGenerator.generate_report()
                            (Q&A + citations)           (Evidence/Interpretation/          (Full Markdown report,
                                                          Recommendation schema)              section by section)
```

The Streamlit UI (`streamlit_app.py`) is a thin presentation layer over these modules.

## 3. Features

- **Local PDF ingestion** with robust handling of empty, corrupted, scanned, and
  duplicate PDFs.
- **Configurable chunking** (size, overlap) with full source metadata preserved on
  every chunk (`source`, `page`, `document_type`, `chunk_index`, `chunk_hash`).
- **Local embeddings** via Sentence-Transformers — no embedding API call ever leaves
  your machine.
- **Persistent ChromaDB** vector store with initial indexing, incremental indexing
  (skips already-embedded chunks by content hash), and full rebuild.
- **Semantic search** returning passages with source, page, and a relevance score.
- **Evidence-grounded RAG Q&A** — the prompt explicitly forbids invented statistics,
  findings, or citations, and requires the model to say when the evidence doesn't
  answer the question.
- **Consumer Insight Engine** producing a structured schema (title, consumer
  problem, evidence, interpretation, sources, confidence, business implication,
  recommended action, potential opportunity) — evidence and interpretation are
  always kept separate, and recommendations are never presented as findings.
- **Automated report generation** producing a full Markdown report with all required
  sections, retrieving fresh context per section (the entire 500+ page corpus is
  never sent to the LLM at once).
- **Streamlit UI** with Dashboard, Document Ingestion, Semantic Search, Ask the
  Research Assistant, Generate Insights, and Generate Report tabs.
- **Graceful Ollama-unavailable handling** everywhere the LLM is called.

## 4. Technology Stack

| Layer            | Technology                                                        |
|-------------------|--------------------------------------------------------------------|
| LLM inference     | [Ollama](https://ollama.com) running **Mistral** locally           |
| RAG orchestration | LangChain (`langchain`, `langchain-core`, `langchain-text-splitters`, `langchain-ollama`, `langchain-huggingface`, `langchain-chroma`) |
| Embeddings        | Sentence-Transformers (`sentence-transformers/all-MiniLM-L6-v2` by default) |
| Vector database   | ChromaDB (persisted on disk)                                       |
| PDF parsing       | `pypdf`                                                             |
| UI                | Streamlit                                                           |
| Config            | `python-dotenv` + a typed `Settings` dataclass                     |
| Testing           | `pytest`, `pytest-mock`                                            |

## 5. Why Local LLM / RAG?

Consumer research documents are frequently sensitive or under NDA. A local-first
architecture means:

- Documents, retrieved passages, questions, and generated insights **never leave the
  machine** — no OpenAI/Gemini/cloud API call is made for inference or embeddings.
- No API key is required to run the core application.
- Retrieval (RAG) keeps the 500+ page corpus out of the LLM's context window —
  only the top-k most relevant chunks are ever sent to Mistral for a given
  question, insight, or report section.

## 6. Installation

### 6.1 Create and activate a virtual environment

**Python 3.11 is required** (pinned in `.python-version`). The pinned
dependency set does not install on Python 3.13+ (no `numpy==1.26.4` wheels).

```bash
python3.11 -m venv .venv
```

Or with [uv](https://docs.astral.sh/uv/), which reads `.python-version`:

```bash
uv venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 6.2 Install dependencies

```bash
pip install -r requirements.txt
```

> The first install will download the Sentence-Transformers embedding model
> weights (a few hundred MB) the first time `EMBEDDING_MODEL` is used — this
> happens locally via Hugging Face's model cache, not through this app's network
> calls to a cloud LLM.

## 7. Ollama Setup

1. Install Ollama from [ollama.com](https://ollama.com).
2. Start the Ollama server (it usually starts automatically after installation; if
   not, run):
   ```bash
   ollama serve
   ```
3. Verify it's running and see installed models:
   ```bash
   ollama list
   ```

## 8. Mistral Setup

If Mistral isn't already installed:

```bash
ollama pull mistral
```

Test it directly:

```bash
ollama run mistral
```

**If your installed Mistral tag is different** (e.g. `mistral:7b-instruct` instead of
`mistral:latest`), set `OLLAMA_MODEL` in your `.env` file to match exactly what
`ollama list` shows, e.g.:

```
OLLAMA_MODEL=mistral:7b-instruct
```

## 9. Embedding Model Setup

No separate setup is required — `sentence-transformers/all-MiniLM-L6-v2` downloads
automatically the first time it's used and then runs entirely on your CPU (or GPU if
`EMBEDDING_DEVICE=cuda` and you have a compatible GPU + PyTorch build). To use a
different local embedding model, change `EMBEDDING_MODEL` in `.env` to any
Sentence-Transformers-compatible model name.

## 10. Environment Variables

Copy the example file and adjust as needed:

```bash
cp .env.example .env
```

| Variable                | Default                                            | Purpose |
|--------------------------|-----------------------------------------------------|---------|
| `OLLAMA_BASE_URL`        | `http://localhost:11434`                            | Ollama server URL |
| `OLLAMA_MODEL`           | `mistral`                                            | Model tag to use for generation |
| `OLLAMA_TEMPERATURE`     | `0.1`                                                | Generation temperature |
| `OLLAMA_REQUEST_TIMEOUT` | `120`                                                | Seconds to wait for Ollama to connect or produce its next output before giving up (covers a hung server or a slow model load; it does not cap total generation time, since output is streamed) |
| `EMBEDDING_MODEL`        | `sentence-transformers/all-MiniLM-L6-v2`             | Local embedding model |
| `EMBEDDING_DEVICE`       | `cpu`                                                | `cpu` or `cuda` |
| `EMBEDDING_BATCH_SIZE`   | `32`                                                  | Batch size for embedding generation |
| `CHUNK_SIZE`             | `1000`                                               | Characters per chunk |
| `CHUNK_OVERLAP`          | `150`                                                | Overlap between chunks |
| `RETRIEVAL_K`            | `5`                                                   | Default number of chunks retrieved per query |
| `MAX_CONTEXT_CHARS`      | `12000`                                              | Cap on context length sent to the LLM per call |
| `DATA_DIR`               | `data/raw`                                           | Where to place source PDFs |
| `VECTORSTORE_DIR`        | `vectorstore`                                        | ChromaDB persistence directory |
| `CHROMA_COLLECTION`      | `fintech_research`                                   | Chroma collection name |

## 11. Adding Research PDFs

Place PDF files into `data/raw/` (or upload them via the **Document Ingestion** tab
in the Streamlit app, which saves them into the same directory):

```bash
cp ~/Downloads/consumer_payments_survey.pdf data/raw/
```

## 12. Building the Index

From the **Document Ingestion** tab in the UI, click **"Build / update index
(incremental)"** to embed only new documents, or **"Rebuild index from scratch"** to
re-embed everything (use this after changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or
`EMBEDDING_MODEL`).

You can also drive ingestion from Python directly:

```python
from app.ingestion import PDFIngestor
from app.chunking import DocumentChunker
from app.vector_store import VectorStoreManager

pages, report = PDFIngestor().ingest_directory()
docs = DocumentChunker().chunk_pages(pages)
VectorStoreManager().add_documents(docs)
print(report)
```

## 13. Running Streamlit

```bash
streamlit run streamlit_app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## 14. Example Research Questions

- "What factors influence digital wallet adoption?"
- "What are the main barriers to mobile payment adoption?"
- "How does trust affect consumer willingness to use digital payments?"
- "What differences exist between younger and older consumers?"
- "What role does transaction fee transparency play in payment method choice?"

## 15. Running Tests

Unit tests (no Ollama required — the LLM is mocked):

```bash
pytest -m "not integration" -v
```

Optional integration test (requires Ollama running with the configured model
installed):

```bash
pytest tests/test_integration_ollama.py -m integration -v
```

## 16. Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard shows "Ollama connection: Unreachable" | Start Ollama (`ollama serve`) and confirm `OLLAMA_BASE_URL` matches where it's running. |
| Dashboard shows the model is not available | Run `ollama list` to see installed models, then `ollama pull mistral` or set `OLLAMA_MODEL` to match your installed tag. |
| "No relevant passages found" in Search | Build the index first from the **Document Ingestion** tab. |
| A PDF produced 0 pages/chunks | It's likely a scanned image PDF with no text layer; this app does not perform OCR. Run it through an OCR tool first, then re-add it. |
| Embedding step is slow on first run | The Sentence-Transformers model weights are downloading; subsequent runs use the local cache. |
| Answers say "does not contain enough information" | This is expected/correct behavior when the corpus doesn't cover the question — it means the grounding prompt is working. |

## 17. Limitations

- Retrieval quality depends on the embedding model and chunking configuration; very
  short or very technical passages may retrieve imperfectly with the default
  MiniLM model.
- No OCR is performed on scanned PDFs — text must already be extractable.
- The relevance score shown is a simple normalized-distance approximation for
  display/ranking, not a calibrated probability.
- Report generation makes multiple sequential local LLM calls (one per section),
  so full reports can take a few minutes depending on your hardware.

## 18. Future Improvements

- Optional OCR pipeline (e.g. `pytesseract`) for scanned documents.
- Hybrid retrieval (BM25 + semantic) for improved recall on keyword-heavy queries.
- Conversation memory for multi-turn research assistant sessions.
- PDF/DOCX export of generated reports using local libraries.
- Swap-in support for other local models served by Ollama (e.g. Llama, Qwen) via
  `OLLAMA_MODEL` with no code changes.

## 19. Security / Privacy

- No PDFs, retrieved passages, questions, or generated content are sent to any cloud
  LLM API (OpenAI, Gemini, or otherwise).
- No API key is required or used by the core application.
- All inference happens through your local Ollama server; all embeddings happen
  through a locally loaded Sentence-Transformers model.
- Logging avoids printing full document contents; only counts, file names, and page
  numbers are logged by default.

## 20. Project Structure

```
fintech-consumer-insight-engine/
│
├── app/
│   ├── __init__.py
│   ├── config.py          # environment-driven Settings
│   ├── ingestion.py        # PDF ingestion + duplicate/error handling
│   ├── chunking.py         # RecursiveCharacterTextSplitter wrapper
│   ├── embeddings.py       # local HuggingFace/Sentence-Transformers embeddings
│   ├── vector_store.py     # ChromaDB persistence, incremental indexing
│   ├── retriever.py        # semantic search over the vector store
│   ├── llm.py               # Ollama chat model + connection health check
│   ├── rag_chain.py         # retrieval -> grounded prompt -> Mistral -> answer
│   ├── insights.py          # structured Consumer Insight Engine
│   ├── reporting.py         # section-by-section Markdown report generation
│   ├── prompts.py           # evidence-grounding prompt templates
│   └── utils.py             # logging, hashing, text cleaning
│
├── data/
│   ├── raw/                 # put source PDFs here
│   └── processed/
│
├── vectorstore/             # persisted ChromaDB collection
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_prompts.py
│   ├── test_insights.py
│   ├── test_vector_store.py
│   └── test_integration_ollama.py   # optional, requires live Ollama
│
├── .env.example
├── .gitignore
├── requirements.txt
├── pytest.ini
├── README.md
└── streamlit_app.py
```
