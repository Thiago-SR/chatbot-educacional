from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_load_pdf_pages_returns_page_numbers_and_text(tmp_path: Path, make_pdf):
    from app.ingestion.pdf_loader import load_pdf_pages

    pdf_path = tmp_path / "sample.pdf"
    make_pdf(pdf_path, ["Page 1 text", "Page 2 text"])

    pages = load_pdf_pages(pdf_path)

    assert [pno for pno, _ in pages] == [1, 2]
    assert all(text for _, text in pages)
    assert any("Page 1 text" in text for _, text in pages)
    assert any("Page 2 text" in text for _, text in pages)


@pytest.mark.unit
def test_chunker_token_windows_overlap_and_majority_page(
    monkeypatch, dummy_tokenizer, tmp_path: Path
):
    import app.ingestion.chunker as chunker_module

    # Avoid Hugging Face tokenizer downloads.
    monkeypatch.setattr(chunker_module, "_get_tokenizer", lambda: dummy_tokenizer)

    # page 1 has 6 tokens: a..f
    # page 2 has 4 tokens: g..j
    pages = [(1, "a b c d e f"), (2, "g h i j")]

    # size=6, overlap=2 => stride=4
    chunks = chunker_module.chunk_pages(pages, chunk_size=6, chunk_overlap=2)

    assert len(chunks) == 2
    assert [c.chunk_index for c in chunks] == [0, 1]

    # window 0: tokens 0..5 => all from page 1
    assert chunks[0].page == 1
    assert len(chunks[0].content.split()) == 6

    # window 1: tokens 4..9 => 2 from page 1, 4 from page 2 => majority page 2
    assert chunks[1].page == 2
    assert len(chunks[1].content.split()) == 6

