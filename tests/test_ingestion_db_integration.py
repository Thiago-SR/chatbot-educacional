from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_ingestion_persists_documents_and_chunks(
    tmp_path: Path,
    monkeypatch,
    make_pdf,
    dummy_tokenizer,
):
    # Import after conftest sets required env vars.
    from app.db.pgvector_setup import setup as setup_db
    import app.ingestion.chunker as chunker_module
    from app.ingestion.pdf_loader import load_pdf_pages
    from scripts import run_ingestion as run_ingestion_module

    # Make tokenization deterministic and avoid Hugging Face downloads.
    monkeypatch.setattr(chunker_module, "_get_tokenizer", lambda: dummy_tokenizer)

    # Prepare DB (extension + tables + index).
    await setup_db()

    # Create a temporary folder with a sample PDF.
    folder = tmp_path / "pdfs"
    folder.mkdir(parents=True, exist_ok=True)

    pdf_filename = f"sample_{uuid.uuid4().hex}.pdf"
    pdf_path = folder / pdf_filename

    # Generate a page with many whitespace-separated tokens.
    # The exact chunk count depends on extracted text, so we compute expected
    # chunks from the loader output using the same tokenizer mock.
    tokens = [f"w{i}" for i in range(120)]
    make_pdf(pdf_path, [" ".join(tokens)])

    pages = load_pdf_pages(pdf_path)
    expected_chunks = chunker_module.chunk_pages(pages)
    expected_count = len(expected_chunks)
    assert expected_count > 0

    # Mock embeddings: return deterministic 384-dim vectors.
    def dummy_encode_texts(texts: list[str], *args, **kwargs) -> list[list[float]]:
        return [[float(i)] * 384 for i in range(len(texts))]

    monkeypatch.setattr(run_ingestion_module, "encode_texts", dummy_encode_texts)

    # Run ingestion end-to-end for that folder.
    exit_code = await run_ingestion_module.run_ingestion(
        folder, "*.pdf", dry_run=False
    )
    assert exit_code == 0

    # Validate persistence in Postgres.
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    try:
        async with engine.connect() as conn:
            doc = (
                await conn.execute(
                    text(
                        "SELECT id, total_chunks FROM documents WHERE filename = :fn"
                    ),
                    {"fn": pdf_filename},
                )
            ).mappings().first()

            assert doc is not None

            document_id = doc["id"]
            assert doc["total_chunks"] == expected_count

            n_chunks = (
                await conn.execute(
                    text(
                        "SELECT COUNT(*) AS c FROM chunks WHERE document_id = :did"
                    ),
                    {"did": document_id},
                )
            ).mappings().first()["c"]

            assert n_chunks == expected_count

            n_embedded = (
                await conn.execute(
                    text(
                        "SELECT COUNT(*) AS c FROM chunks WHERE document_id = :did AND embedding IS NOT NULL"
                    ),
                    {"did": document_id},
                )
            ).mappings().first()["c"]

            assert n_embedded == expected_count

            # chunk_index should be sequential from 0..N-1.
            rows = (
                await conn.execute(
                    text(
                        "SELECT chunk_index FROM chunks WHERE document_id = :did ORDER BY chunk_index"
                    ),
                    {"did": document_id},
                )
            ).fetchall()
            assert [r[0] for r in rows] == list(range(expected_count))
    finally:
        await engine.dispose()

    # Cleanup DB rows for this test.
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM documents WHERE filename = :fn"),
                {"fn": pdf_filename},
            )
    finally:
        await engine.dispose()

