# Offline-RAG-Document-QA

A local, offline Retrieval-Augmented Generation system for asking questions
about your own documents (PDF, TXT, Markdown, DOCX). No data leaves your
machine — embeddings, vector search, and generation all run locally via
[Ollama](https://ollama.com).

## Architecture

```
                 ┌──────────────┐
   documents ──► │  ingest.py    │  load → clean → chunk (+ metadata)
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │   rag.py      │  embed (sentence-transformers)
                 │               │  store (ChromaDB, persistent)
                 └──────┬───────┘
                        ▼  (on a question)
                 ┌──────────────┐
                 │   rag.py      │  embed query → top-k similarity search
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │ prompts.py    │  build grounded prompt + context + history
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │   Ollama      │  local LLM generates the answer
                 └──────┬───────┘
                        ▼
                 answer + cited sources (Streamlit UI)
```

## Installation

1. **Clone and enter the project**
   ```bash
   git clone <your-repo-url> rag-doc-qa
   cd rag-doc-qa
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Copy the environment template**
   ```bash
   cp .env.example .env
   ```

## Running Ollama

1. [Install Ollama](https://ollama.com/download) for your OS.
2. Start the server:
   ```bash
   ollama serve
   ```
3. Pull the default model (in another terminal):
   ```bash
   ollama pull llama3
   ```

## Running the app

```bash
streamlit run streamlit_app.py
```

Then, in the sidebar:
1. Upload your PDF/TXT/MD/DOCX files.
2. Click **Re-index** (or **Full re-index** to rebuild from scratch).
3. Ask questions in the chat box.

### CLI usage (no UI)

```bash
python app.py index            # index new documents
python app.py index --force    # wipe and rebuild the whole index
python app.py ask "What is the notice period in the contract?"
python app.py stats            # show how many chunks are indexed
```

## How ingestion works

- `ingest.py` recursively scans `data/` for supported files.
- Each file is parsed with the appropriate loader (PyMuPDF for PDF, native
  Python for TXT/MD, `python-docx` for DOCX).
- Text is cleaned (whitespace/control-character normalization) and split
  into overlapping chunks (`chunk_size` / `chunk_overlap`, configurable),
  preferring paragraph/sentence boundaries.
- Every chunk carries metadata: `filename`, `page` (PDF only), `doc_type`,
  `chunk_index`, and a SHA-256 `file_hash` of the source file's bytes.
- The `file_hash` is what powers **incremental indexing**: `index_documents()`
  looks up which hashes already exist in Chroma and only embeds files whose
  content hash isn't already stored — so re-running indexing skips
  unchanged files and only processes new or modified ones.

## How retrieval works

- The user's question is embedded with the same `sentence-transformers`
  model used at index time (`all-MiniLM-L6-v2` by default).
- ChromaDB performs a cosine-similarity search over the persistent
  collection and returns the top-k most relevant chunks.
- `prompts.py` assembles those chunks into a numbered context block plus
  recent conversation history, and instructs the model to answer only from
  that context and cite sources like `[Source 1]`.
- The prompt is sent to a local Ollama model via `/api/generate`; the
  response is shown alongside a **Sources** panel with filename/page and
  similarity distance for each retrieved chunk.

## Configuration

Defaults live in `config.yaml`; any value can be overridden by an
environment variable in `.env` named `RAG_<FIELD_NAME_UPPER>`:

| Setting | config.yaml key | Env override | Default |
|---|---|---|---|
| Ollama host | `ollama.ollama_host` | `RAG_OLLAMA_HOST` | `http://localhost:11434` |
| Ollama model | `ollama.ollama_model` | `RAG_OLLAMA_MODEL` | `llama3` |
| Temperature | `ollama.temperature` | `RAG_TEMPERATURE` | `0.2` |
| Embedding model | `embeddings.embedding_model` | `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| Chunk size | `chunking.chunk_size` | `RAG_CHUNK_SIZE` | `800` |
| Chunk overlap | `chunking.chunk_overlap` | `RAG_CHUNK_OVERLAP` | `120` |
| Top-k | `retrieval.top_k` | `RAG_TOP_K` | `4` |
| Collection name | `chroma.collection_name` | `RAG_COLLECTION_NAME` | `documents` |
| Log level | `logging.log_level` | `RAG_LOG_LEVEL` | `INFO` |

Model, temperature, and top-k can also be adjusted live from the Streamlit
sidebar for experimentation without editing files.

## Screenshots

_Add screenshots here after running the app locally:_

- `assets/screenshot-chat.png` — main chat interface
- `assets/screenshot-sidebar.png` — sidebar controls
- `assets/screenshot-sources.png` — source citation panel

## Example queries

- "Summarize the key obligations of the vendor in this contract."
- "What is the refund policy described in section 3?"
- "List all deadlines mentioned across these documents."

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Cannot reach Ollama" | `ollama serve` not running | Start it, or check `RAG_OLLAMA_HOST` |
| Empty/odd answers | Model not pulled | `ollama pull <model-name>` |
| PDF skipped with "no extractable text" | Scanned/image-only PDF | Run OCR first, or use a different source |
| Re-index does nothing | Files already indexed (same content hash) | Use **Full re-index** to force a rebuild |
| Slow first query | Embedding model downloading/loading | Wait for first load; subsequent calls are fast |
| `ModuleNotFoundError` | Dependencies not installed in active env | Re-run `pip install -r requirements.txt` inside your venv |

## Future improvements

- OCR fallback for scanned PDFs.
- Hybrid search (BM25 + embeddings) for better recall on exact terms.
- Streaming token-by-token responses in the UI.
- Multi-user support with per-user collections.
- Support for additional formats (HTML, EPUB, CSV).

## Project structure

```
rag-doc-qa/
├── app.py              # CLI entry point (index / ask / stats)
├── ingest.py            # document loading, cleaning, chunking
├── rag.py                # embeddings, Chroma storage, retrieval, generation
├── config.py             # settings loader (YAML + env overrides)
├── utils.py              # logging, hashing helpers
├── prompts.py             # prompt templates
├── streamlit_app.py       # Streamlit UI
├── requirements.txt
├── config.yaml
├── .env.example
├── .gitignore
├── data/                  # put your source documents here
├── chroma_db/             # persistent vector store (auto-created)
├── logs/                  # rotating log files (auto-created)
└── assets/                # screenshots, static assets
```
