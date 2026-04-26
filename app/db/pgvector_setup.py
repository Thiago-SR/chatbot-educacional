"""
Database setup: enables the pgvector extension and creates all tables.

Run once before starting the application:
    python -m app.db.pgvector_setup
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.db.models import Base

logger = logging.getLogger(__name__)


async def create_pgvector_extension(engine) -> None:
    """Enables the pgvector extension if it does not already exist."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension enabled.")


async def create_tables(engine) -> None:
    """Creates all tables defined in the models."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Tables created successfully.")


async def create_vector_index(engine) -> None:
    """
    Creates an HNSW index on the embedding column for fast similarity searches.
    HNSW is more efficient than IVFFlat for collections up to ~1M vectors.
    """
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        logger.info("HNSW index created on the embedding column.")


async def setup() -> None:
    """Runs the full initial database setup."""
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
    )

    try:
        logger.info("Starting database setup...")
        await create_pgvector_extension(engine)
        await create_tables(engine)
        await create_vector_index(engine)
        logger.info("Database setup completed successfully.")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    asyncio.run(setup())
