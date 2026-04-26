# Educational Chatbot — Afro-Brazilian Culture

An interactive RAG (Retrieval-Augmented Generation) chatbot dedicated to spreading knowledge about Afro-Brazilian culture. The project covers history, traditions, African-rooted religions, music, cuisine, and the contributions of Black Brazilians to the country's identity.

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 16 + pgvector (via Docker) |
| ORM / async | SQLAlchemy 2 + asyncpg |
| Configuration | pydantic-settings |
| LLM (future phase) | Groq API (`llama-3.3-70b-versatile`) |
| Embeddings (future phase) | sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` (384d) |

## Project structure

```
chatbot-educacional/
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI entry point
│   ├── config.py             # Settings via pydantic-settings
│   └── db/
│       ├── __init__.py
│       ├── models.py         # Tables: documents, chunks (vector 384d)
│       └── pgvector_setup.py # Creates extension, tables and HNSW index
├── scripts/
│   └── test_db_connection.py # Validates SELECT 1 + vector extension + tables
├── data/
│   └── documents/            # PDFs for ingestion (future phase)
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup — step by step

### 1. Clone and create the virtual environment

```bash
git clone <repo-url>
cd chatbot-educacional

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` and fill in at least `GROQ_API_KEY` (required for the chat phase; the placeholder value is fine during the environment and DB phases).

### 4. Start the database

```bash
docker compose up -d postgres
```

Wait until the container is healthy (`docker ps` should show `(healthy)`).

### 5. Create the schema (vector extension + tables + HNSW index)

```bash
python -m app.db.pgvector_setup
```

Expected output:
```
pgvector extension enabled.
Tables created successfully.
HNSW index created on the embedding column.
Database setup completed successfully.
```

### 6. Test the connection

```bash
python scripts/test_db_connection.py
```

Expected output:
```
[OK] SELECT 1 returned 1 — connection is working.
[OK] pgvector extension is active.
[OK] Tables found: chunks, documents
```

### 7. Run the API locally

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/` — should return `{"status": "ok", ...}`.  
Interactive docs: `http://127.0.0.1:8000/docs`

### 8. Start everything via Docker Compose (DB + app)

```bash
docker compose up --build
```

## Environment variables

| Variable | Description | Required now? |
|----------|-------------|---------------|
| `DATABASE_URL` | asyncpg connection URL | Yes |
| `POSTGRES_USER` | Postgres user (docker-compose) | Yes |
| `POSTGRES_PASSWORD` | Postgres password (docker-compose) | Yes |
| `POSTGRES_DB` | Database name (docker-compose) | Yes |
| `GROQ_API_KEY` | Groq API key | Chat phase |
| `GROQ_MODEL` | Groq model (default: `llama-3.3-70b-versatile`) | Chat phase |
| `EMBEDDING_MODEL` | sentence-transformers model ID | Ingestion phase |
| `CHUNK_SIZE` | Chunk size in tokens (default: 500) | Ingestion phase |
| `CHUNK_OVERLAP` | Overlap between chunks (default: 50) | Ingestion phase |
| `TOP_K_RESULTS` | Chunks retrieved per query (default: 5) | RAG phase |

## Roadmap

- [x] **Phase 1** — Environment and database (current)
  - Python structure, venv, requirements
  - PostgreSQL + pgvector via Docker Compose
  - `documents` and `chunks` tables with `vector(384)` column
  - SELECT 1 connection test
- [ ] **Phase 2** — Document ingestion
  - Read PDFs from `data/documents/`
  - Chunking and embedding generation (sentence-transformers)
  - Store vectors in pgvector
- [ ] **Phase 3** — Retrieval and RAG
  - Cosine similarity search (HNSW index)
  - Prompt assembly with retrieved context
- [ ] **Phase 4** — Chat with Groq
  - `POST /chat/` endpoint integrated with the RAG + Groq pipeline
