# FinTech Consumer Insight Engine

A **local-first Retrieval-Augmented Generation (RAG)** application that ingests large
volumes of unstructured FinTech / digital-payments research (500+ pages of PDFs),
indexes it into a persistent vector database, and uses a **locally running Mistral
model via Ollama** to answer questions, generate structured consumer insights, and
produce full Markdown research reports. In the default local mode this happens
**without sending any document content, question, or generated output to a cloud LLM
API**.

An optional **hosted mode** (`LLM_PROVIDER=gemini`) uses the Google Gemini API for
generation instead, e.g. for a Streamlit Community Cloud deployment where Ollama is
not available. Hosted mode is **not fully local**: see
[Hosted mode with Gemini](#14-hosted-mode-with-gemini-optional) for exactly what is sent.

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
   **Mistral running locally through Ollama** (default), or the hosted Gemini API if
   `LLM_PROVIDER=gemini` is set.

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
- **Graceful provider-unavailable handling** everywhere the LLM is called, with
  specific messages for an unreachable server, a missing model, a timeout, or a
  misconfigured provider.
- **Optional hosted generation via Gemini** (`LLM_PROVIDER=gemini`), using the same
  prompts, retrieval, citations, and output formats as the local mode.

## 4. Technology Stack

| Layer            | Technology                                                        |
|-------------------|--------------------------------------------------------------------|
| LLM inference     | [Ollama](https://ollama.com) running **Mistral** locally (default); optional hosted [Gemini API](https://ai.google.dev/gemini-api/docs) via `google-genai` |
| RAG orchestration | LangChain (`langchain`, `langchain-core`, `langchain-text-splitters`, `langchain-ollama`, `langchain-huggingface`, `langchain-chroma`) |
| Embeddings        | Sentence-Transformers (`sentence-transformers/all-MiniLM-L6-v2` by default) |
| Vector database   | ChromaDB (persisted on disk)                                       |
| PDF parsing       | `pypdf`                                                             |
| UI                | Streamlit                                                           |
| Config            | `python-dotenv` + a typed `Settings` dataclass                     |
| Testing           | `pytest`, `pytest-mock`                                            |

## 5. Why Local LLM / RAG?

Consumer research documents are frequently sensitive or under NDA. The default
local-first mode (`LLM_PROVIDER=ollama`) means:

- Documents, retrieved passages, questions, and generated insights **never leave the
  machine** — no OpenAI/Gemini/cloud API call is made for inference or embeddings.
- No API key is required to run the application.
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
| `LLM_PROVIDER`           | `ollama`                                             | `ollama` (local) or `gemini` (hosted Google Gemini API) |
| `GEMINI_API_KEY`         | _(empty)_                                            | Gemini API key; required only when `LLM_PROVIDER=gemini`. Never commit it. |
| `GEMINI_MODEL`           | `gemini-3.8-flash`                                   | Gemini model ID, e.g. `gemini-3.5-flash-lite` for lower cost |
| `GEMINI_REQUEST_TIMEOUT` | `120`                                                | Seconds before a Gemini request attempt times out |
| `GEMINI_THINKING_LEVEL`  | model default (`low` in public demo mode)            | `minimal`, `low`, `medium`, or `high`; lower is faster. `gemini-3.8-flash` supports `low`/`medium`/`high` |
| `GEMINI_RETRY_ATTEMPTS`  | `3`                                                  | Total attempts per Gemini request; transient 500/502/503/504 errors, timeouts, and connection errors are retried by the SDK with bounded backoff (about 2s, then 4s) |
| `PUBLIC_DEMO_MODE`       | `false`                                              | Read-only public demo: disables uploads, deletion, and reindexing; rate-limits model calls |
| `DEMO_SESSION_REQUEST_LIMIT` | `20`                                             | Demo mode: model calls per browser session per window (a report counts as 10) |
| `DEMO_GLOBAL_REQUEST_LIMIT`  | `200`                                            | Demo mode: model calls per server process per window, across all visitors |
| `DEMO_RATE_WINDOW_SECONDS`   | `3600`                                           | Demo mode: length of the rate-limit window |
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
| `DATA_DIR`               | `data/raw` (`demo_data` in public demo mode)          | Where to place source PDFs |
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

**Running locally with Ollama** is the default and needs no API key: start Ollama
(section 7), make sure `ollama list` shows your model, leave `LLM_PROVIDER` unset (or
set it to `ollama`), and run the command above.

## 14. Hosted Mode with Gemini (Optional)

Use this when the app runs somewhere Ollama is not available, such as Streamlit
Community Cloud. Selecting Gemini does **not** require an Ollama server.

> **Privacy:** in hosted mode, each generation request sends **the user's question or
> topic and the retrieved document passages** (up to `MAX_CONTEXT_CHARS` per request,
> with their file names and page numbers) to Google's Gemini API. Full PDFs are not
> uploaded, and embeddings and the vector store stay on the server running the app —
> but this mode is **not fully local**. Check that your documents may be shared with
> Google under the [Gemini API terms](https://ai.google.dev/gemini-api/terms) before
> enabling it. The Dashboard's status check only looks up the model's metadata and
> sends no document content.

1. Create an API key in [Google AI Studio](https://aistudio.google.com/apikey).
2. Configure the app:

   **Locally or on your own server** — in `.env` (git-ignored) or the environment:
   ```
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your-key-here
   GEMINI_MODEL=gemini-3.8-flash
   ```

   **On Streamlit Community Cloud** — open your app's **Settings → Secrets** and add:
   ```toml
   LLM_PROVIDER = "gemini"
   GEMINI_API_KEY = "your-key-here"
   GEMINI_MODEL = "gemini-3.8-flash"
   ```
   For local testing of secrets, copy `.streamlit/secrets.toml.example` to
   `.streamlit/secrets.toml` (git-ignored). Environment variables take precedence
   over Streamlit secrets.

3. Open the Dashboard: it should show "Gemini API: Connected". A rejected key, an
   unavailable model, or an exceeded quota is reported there and in each workflow.

The API key is never logged or displayed, and is excluded from the settings object's
`repr`. Model IDs change over time; see the
[Gemini models page](https://ai.google.dev/gemini-api/docs/models) for current ones.

### Public demo mode

For a public deployment (e.g. Streamlit Community Cloud), set `PUBLIC_DEMO_MODE=true`
(in Streamlit secrets or the environment). It is off by default, so local development
is unaffected. In demo mode:

- **Bundled demo corpus.** Unless `DATA_DIR` is set explicitly, documents are read from
  `demo_data/`, which contains two short, clearly labelled *illustrative* PDFs
  (no real survey data). On start-up, if the vector index is empty, the app indexes
  them automatically with the normal ingestion → chunking → local embedding pipeline
  (`seed_demo_index`). Seeding runs once per server process, is add-only, and never
  modifies an index that already has content; it does nothing when demo mode is off.
  Because Streamlit Community Cloud can reset the filesystem, the committed PDFs are
  the source of truth and the index is recreated after a reset (the first visitor
  afterwards waits for indexing, and the embedding model downloads on first use).
  Regenerate the PDFs with `python scripts/build_demo_corpus.py`.
- **Read-only documents.** The Knowledge Base shows the indexed collection but hides
  upload, delete, and reindex controls, and the indexing code refuses to run
  (`build_index` raises `DemoModeRestriction`).
- **Research stays available:** Dashboard, Semantic Search, Research Assistant,
  Insights Studio, and Report Builder.
- **Rate-limited model calls.** Each browser session may make
  `DEMO_SESSION_REQUEST_LIMIT` calls, and the whole server process
  `DEMO_GLOBAL_REQUEST_LIMIT` calls, per `DEMO_RATE_WINDOW_SECONDS`. An answer or
  insight costs 1 call; a report costs one per section (10). Semantic Search makes
  no model calls and is not limited.
- **Privacy notice.** Every page shows a demo banner; with Gemini it states that
  questions and retrieved passages are sent to Google's Gemini API.

Limitations: the limits are in memory, so they reset when the app restarts and are
per process (not shared across replicas). The per-session limit can be bypassed by
reloading the page, which starts a new session; the process-wide limit still applies.
There is no per-IP or per-user tracking and no authentication — set a budget or quota
on the Gemini API key in Google AI Studio as the hard cost ceiling.

## 15. Example Research Questions

- "What factors influence digital wallet adoption?"
- "What are the main barriers to mobile payment adoption?"
- "How does trust affect consumer willingness to use digital payments?"
- "What differences exist between younger and older consumers?"
- "What role does transaction fee transparency play in payment method choice?"

## 16. Running Tests

Unit tests (no Ollama or Gemini API key required — the LLMs are mocked and no live API
calls are made):

```bash
pytest -m "not integration" -v
```

Optional integration test (requires Ollama running with the configured model
installed):

```bash
pytest tests/test_integration_ollama.py -m integration -v
```

## 17. Troubleshooting

| Symptom | Fix |
|---|---|
| "LLM_PROVIDER is set to 'gemini' but GEMINI_API_KEY is not set" | Add the key to `.env`/the environment, or to Streamlit secrets on Community Cloud. |
| "Gemini rejected the API key" | Check the key in Google AI Studio and that it has no extra spaces. |
| "Gemini model ... was not found" | Set `GEMINI_MODEL` to a current model ID from the Gemini models page. |
| "Unsupported LLM_PROVIDER" | Set `LLM_PROVIDER` to `ollama` or `gemini`. |
| Dashboard shows "Ollama connection: Unreachable" | Start Ollama (`ollama serve`) and confirm `OLLAMA_BASE_URL` matches where it's running. |
| Dashboard shows the model is not available | Run `ollama list` to see installed models, then `ollama pull mistral` or set `OLLAMA_MODEL` to match your installed tag. |
| "No relevant passages found" in Search | Build the index first from the **Document Ingestion** tab. |
| A PDF produced 0 pages/chunks | It's likely a scanned image PDF with no text layer; this app does not perform OCR. Run it through an OCR tool first, then re-add it. |
| Embedding step is slow on first run | The Sentence-Transformers model weights are downloading; subsequent runs use the local cache. |
| Answers say "does not contain enough information" | This is expected/correct behavior when the corpus doesn't cover the question — it means the grounding prompt is working. |

## 18. Limitations

- Retrieval quality depends on the embedding model and chunking configuration; very
  short or very technical passages may retrieve imperfectly with the default
  MiniLM model.
- No OCR is performed on scanned PDFs — text must already be extractable.
- The relevance score shown is a simple normalized-distance approximation for
  display/ranking, not a calibrated probability.
- Report generation makes sequential LLM calls: 10 in **Detailed** mode (one per
  section) or 3 in **Fast** mode (related sections grouped, each group with its own
  section-specific retrieval), so reports can take a while depending on your hardware
  or Gemini latency. If a Fast-mode group fails, the rest of the report is kept, the
  failed sections are marked as not generated, and only that group can be retried.

## 19. Future Improvements

- Optional OCR pipeline (e.g. `pytesseract`) for scanned documents.
- Hybrid retrieval (BM25 + semantic) for improved recall on keyword-heavy queries.
- Conversation memory for multi-turn research assistant sessions.
- PDF/DOCX export of generated reports using local libraries.
- Swap-in support for other local models served by Ollama (e.g. Llama, Qwen) via
  `OLLAMA_MODEL` with no code changes.

## 20. Security / Privacy

**Local mode (`LLM_PROVIDER=ollama`, the default):**

- No PDFs, retrieved passages, questions, or generated content are sent to any cloud
  LLM API (OpenAI, Gemini, or otherwise).
- No API key is required or used.
- All inference happens through your local Ollama server.

**Hosted mode (`LLM_PROVIDER=gemini`):**

- Questions/topics and the retrieved document passages used for each generation
  request (with file names and page numbers) are sent to Google's Gemini API, and
  Gemini's responses come back over the network. This mode is not fully local.
- Full PDFs, the vector store, and embeddings are never sent.
- `GEMINI_API_KEY` is read from the environment, `.env`, or Streamlit secrets; it is
  never hardcoded, logged, or displayed. `.env` and `.streamlit/secrets.toml` are
  git-ignored.

**Both modes:**

- All embeddings are computed by a locally loaded Sentence-Transformers model, and
  ChromaDB telemetry is disabled.
- Logging avoids printing full document contents; only counts, file names, and page
  numbers are logged by default.

## 21. Project Structure

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
│   ├── llm.py               # provider selection, health checks, error messages
│   ├── gemini.py            # optional hosted Gemini adapter (google-genai)
│   ├── indexing.py          # ingest -> chunk -> sync the vector store
│   ├── rag_chain.py         # retrieval -> grounded prompt -> Mistral -> answer
│   ├── insights.py          # structured Consumer Insight Engine
│   ├── reporting.py         # section-by-section Markdown report generation
│   ├── prompts.py           # evidence-grounding prompt templates
│   ├── ui.py                # Streamlit rendering helpers + scoped CSS
│   ├── research_session.py  # session-scoped research history + saved queries
│   └── utils.py             # logging, hashing, text cleaning
│
├── data/
│   ├── raw/                 # put source PDFs here
│   └── processed/
│
├── demo_data/               # illustrative demo PDFs (public demo mode)
├── scripts/build_demo_corpus.py  # regenerates demo_data/*.pdf
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
│   ├── test_index_sync.py
│   ├── test_uploads.py
│   ├── test_llm.py
│   ├── test_providers.py
│   └── test_integration_ollama.py   # optional, requires live Ollama
│
├── .streamlit/config.toml           # theme; Streamlit usage stats disabled
├── .streamlit/secrets.toml.example
├── .env.example
├── .gitignore
├── requirements.txt
├── pytest.ini
├── README.md
└── streamlit_app.py
```
