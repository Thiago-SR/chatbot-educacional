"""
CLI: ingest PDFs from a folder into documents + chunks tables.

Duplicate filenames: existing document (and chunks via CASCADE) are deleted
and the PDF is re-ingested.

Usage (from project root, venv activated, .env filled in):
    python scripts/run_ingestion.py --folder "C:\\path\\to\\pdfs"

Verify in Postgres after a successful run:
    SELECT filename, total_chunks FROM documents ORDER BY filename;
    SELECT COUNT(*) AS chunk_count FROM chunks;

Prerequisites:
    python -m app.db.pgvector_setup
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.db.models import Chunk, Document
from app.ingestion import chunk_pages, encode_texts, load_pdf_pages

logger = logging.getLogger(__name__)


async def _delete_existing_by_filename(session: AsyncSession, filename: str) -> bool:
    result = await session.execute(
        select(Document).where(Document.filename == filename)
    )
    existing = result.scalar_one_or_none()
    if existing is None:
        return False
    await session.delete(existing)
    await session.flush()
    logger.info("Removed existing document for re-ingestion: %s", filename)
    return True


async def ingest_pdf(
    session: AsyncSession | None,
    pdf_path: Path,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Process one PDF. Returns (page_count, chunk_count).
    Does not commit; caller commits per file.
    """
    filename = pdf_path.name
    pages = load_pdf_pages(pdf_path)
    text_chunks = chunk_pages(pages)

    if dry_run:
        logger.info(
            "[dry-run] %s — pages=%d chunks=%d",
            filename,
            len(pages),
            len(text_chunks),
        )
        return len(pages), len(text_chunks)

    if session is None:
        raise ValueError("Database session is required for non-dry-run ingestion")
    await _delete_existing_by_filename(session, filename)

    t0 = time.perf_counter()
    contents = [c.content for c in text_chunks]
    embeddings = encode_texts(contents)
    elapsed = time.perf_counter() - t0

    if len(embeddings) != len(text_chunks):
        raise RuntimeError(
            f"Embedding count mismatch for {filename}: "
            f"{len(embeddings)} vs {len(text_chunks)}"
        )

    doc = Document(
        filename=filename,
        title=pdf_path.stem,
        total_chunks=len(text_chunks),
    )
    session.add(doc)
    await session.flush()

    for text_chunk, vector in zip(text_chunks, embeddings):
        session.add(
            Chunk(
                document_id=doc.id,
                content=text_chunk.content,
                embedding=vector,
                chunk_index=text_chunk.chunk_index,
                page=text_chunk.page,
            )
        )

    logger.info(
        "Ingested %s — pages=%d chunks=%d embedding_time=%.1fs",
        filename,
        len(pages),
        len(text_chunks),
        elapsed,
    )
    return len(pages), len(text_chunks)


async def run_ingestion(
    folder: Path,
    pattern: str,
    *,
    dry_run: bool,
) -> int:
    if not folder.is_dir():
        logger.error("Folder not found: %s", folder)
        return 1

    pdf_files = sorted(folder.glob(pattern))
    if not pdf_files:
        logger.warning("No files matching %s in %s", pattern, folder)
        return 0

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    total_pages = 0
    total_chunks = 0
    errors = 0

    try:
        for pdf_path in pdf_files:
            try:
                if dry_run:
                    pages, chunks = await ingest_pdf(None, pdf_path, dry_run=True)
                else:
                    async with session_factory() as session:
                        pages, chunks = await ingest_pdf(
                            session, pdf_path, dry_run=False
                        )
                        await session.commit()
                total_pages += pages
                total_chunks += chunks
            except (OSError, RuntimeError, ValueError):
                logger.exception("Failed to ingest %s", pdf_path.name)
                errors += 1
    finally:
        await engine.dispose()

    logger.info(
        "Done — files=%d pages=%d chunks=%d errors=%d dry_run=%s",
        len(pdf_files),
        total_pages,
        total_chunks,
        errors,
        dry_run,
    )
    if not dry_run and errors == 0:
        logger.info(
            "Verify: SELECT filename, total_chunks FROM documents; "
            "SELECT COUNT(*) FROM chunks;"
        )
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into documents and chunks tables.",
        epilog=(
            "Duplicate filenames: existing rows are deleted and re-ingested. "
            "After success, verify with: "
            "SELECT filename, total_chunks FROM documents; "
            "SELECT COUNT(*) FROM chunks;"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="Glob pattern for PDF files (default: *.pdf)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk only; do not write to the database",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    exit_code = asyncio.run(
        run_ingestion(args.folder, args.pattern, dry_run=args.dry_run)
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
