"""
Extract plain text from each page of a PDF using PyMuPDF.
"""

import re
from pathlib import Path

import fitz


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """
    Returns a list of (page_number, text) tuples.
    page_number is 1-based, matching PDF page labels.
    """
    path = Path(pdf_path)
    pages: list[tuple[int, str]] = []

    with fitz.open(path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            raw = page.get_text("text") or ""
            pages.append((page_index + 1, _normalize_text(raw)))

    return pages
