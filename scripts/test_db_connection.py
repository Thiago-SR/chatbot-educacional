"""
Database connection test.

Checks:
  1. SELECT 1  — confirms the database is reachable
  2. pgvector extension is active
  3. Lists tables created in the public schema

Usage (from the project root, with venv activated and .env filled in):
    python scripts/test_db_connection.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings


async def test() -> bool:
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    ok = True

    try:
        async with engine.connect() as conn:
            # 1. SELECT 1
            result = await conn.execute(text("SELECT 1"))
            value = result.scalar()
            if value == 1:
                print("[OK] SELECT 1 returned 1 — connection is working.")
            else:
                print(f"[FAIL] SELECT 1 returned unexpected value: {value}")
                ok = False

            # 2. pgvector extension
            result = await conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            ext = result.scalar()
            if ext == "vector":
                print("[OK] pgvector extension is active.")
            else:
                print("[WARN] pgvector extension NOT found. Run first: python -m app.db.pgvector_setup")
                ok = False

            # 3. Tables in the public schema
            result = await conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' ORDER BY tablename"
                )
            )
            tables = [row[0] for row in result.fetchall()]
            if tables:
                print(f"[OK] Tables found: {', '.join(tables)}")
            else:
                print("[WARN] No tables found. Run: python -m app.db.pgvector_setup")

    except Exception as exc:
        print(f"[FAIL] Could not connect: {exc}")
        ok = False
    finally:
        await engine.dispose()

    return ok


if __name__ == "__main__":
    success = asyncio.run(test())
    sys.exit(0 if success else 1)
