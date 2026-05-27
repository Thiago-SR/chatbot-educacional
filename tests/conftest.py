import os
from collections.abc import Callable
from pathlib import Path

import fitz  # PyMuPDF
import pytest


# ---------------------------------------------------------------------------
# Test environment defaults
# ---------------------------------------------------------------------------
#
# app/config.py instantiates Settings at import time and GROQ_API_KEY is required.
# Ensure it exists before any app imports happen.
os.environ.setdefault("GROQ_API_KEY", "test")

# Make chunking faster/deterministic for tests.
os.environ.setdefault("CHUNK_SIZE", "50")
os.environ.setdefault("CHUNK_OVERLAP", "10")


class DummyTokenizer:
    """
    Simple whitespace tokenizer with deterministic encode/decode.

    - encode(): split by whitespace, return [1..n]
    - decode(): map ids back to strings "tok{id}"
    """

    # Used by app.ingestion.chunker._encode_full_text
    model_max_length = 512

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        parts = [p for p in text.split() if p.strip()]
        return list(range(1, len(parts) + 1))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{tid}" for tid in token_ids)


@pytest.fixture
def dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture
def make_pdf() -> Callable[[Path, list[str]], None]:
    """
    Create a temporary PDF with 1 page per entry in `pages_text`.
    """

    def _make_pdf(pdf_path: Path, pages_text: list[str]) -> None:
        doc = fitz.open()
        try:
            for page_text in pages_text:
                page = doc.new_page()
                page.insert_text((72, 72), page_text, fontsize=12)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(pdf_path))
        finally:
            doc.close()

    return _make_pdf

