# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-powered educational chatbot about Afro-Brazilian culture. Python 3.12, FastAPI, PostgreSQL 16 with pgvector, async SQLAlchemy 2, sentence-transformers embeddings (384-dim), Groq LLM (future phase).

## Commands

### Database setup (required before first run)
```bash
docker compose up -d postgres          # start PostgreSQL + pgvector
python -m app.db.pgvector_setup        # create extension, tables, HNSW index
```

### Run the API
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Run everything via Docker
```bash
docker compose up --build              # entrypoint.sh auto-runs schema setup
```

### PDF ingestion
```bash
python scripts/run_ingestion.py --folder "path/to/pdfs"
python scripts/run_ingestion.py --folder "path/to/pdfs" --dry-run  # preview only
```

### Tests
```bash
pip install -r requirements-dev.txt    # install test dependencies
pytest -m unit                         # fast, no DB needed, uses dummy tokenizer
pytest -m integration                  # requires running Postgres with schema created
pytest                                 # all tests
```

### Manual retriever test (requires ingested data)
```bash
python tests/test_retriever_manual.py
```

## Architecture

### Pipeline flow
`PDF files` → `app.ingestion` (extract → chunk → embed) → `PostgreSQL/pgvector` → `app.rag` (embed query → retrieve) → (future: Groq LLM response)

### Key packages

- **`app/embedder.py`** — Unified sentence-transformer embeddings. Single source of truth for both ingestion and query paths. All vectors are normalized for cosine similarity. Both `app/ingestion/embedder.py` and `app/rag/embedder.py` re-export from here.

- **`app/ingestion/`** — PDF-to-database pipeline. Three stages exposed via `__init__.py`:
  - `pdf_loader.load_pdf_pages()` — PyMuPDF text extraction, returns `(page_number, text)` tuples
  - `chunker.chunk_pages()` — sliding token windows with overlap, uses HuggingFace tokenizer aligned to the embedding model
  - `embedder.encode_texts()` — batch embedding (re-exported from `app.embedder`)

- **`app/rag/`** — Query-time retrieval:
  - `retriever.Retriever` — cosine similarity search against pgvector using raw SQL with `<=>` operator. Validates vector dimensions (384) and finite values.

- **`app/db/`** — Database layer:
  - `models.py` — SQLAlchemy ORM: `Document` (1) → (many) `Chunk` with `Vector(384)` column
  - `session.py` — async engine with connection pooling (`pool_pre_ping`, `pool_size=10`, `max_overflow=20`)
  - `pgvector_setup.py` — schema bootstrap (extension, tables, HNSW index) + legacy column migration

- **`app/config.py`** — `pydantic-settings` singleton (`settings`), loads from `.env`. `GROQ_API_KEY` is optional until the chat phase.

- **`app/main.py`** — FastAPI app with lifespan (engine disposal on shutdown), CORS middleware, `/health` endpoint.

### Test conventions
- Tests use `@pytest.mark.unit` and `@pytest.mark.integration` markers (defined in `pytest.ini`)
- `conftest.py` sets small chunk params before any app imports
- Unit tests monkeypatch `_get_tokenizer` with a `DummyTokenizer` (whitespace-based) to avoid HuggingFace downloads
- Integration tests monkeypatch `encode_texts` with deterministic 384-dim vectors
- `make_pdf` fixture creates temporary PDFs via PyMuPDF for testing

## Environment

Copy `.env.example` to `.env`. Key settings: `CHUNK_SIZE` (default 500), `CHUNK_OVERLAP` (default 50), `TOP_K_RESULTS` (default 5). `GROQ_API_KEY` is optional until the chat phase is implemented.
