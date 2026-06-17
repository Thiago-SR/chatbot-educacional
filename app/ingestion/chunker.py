"""
Split document text into overlapping token windows aligned with the embedding model tokenizer.
"""

from dataclasses import dataclass
from functools import lru_cache

from transformers import AutoTokenizer

from app.config import settings


@dataclass(frozen=True)
class TextChunk:
    content: str
    page: int | None
    chunk_index: int


@lru_cache(maxsize=1)
def _get_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)


def _encode_full_text(tokenizer: AutoTokenizer, text: str) -> list[int]:
    """
    Tokenize an entire page without truncating at model_max_length (512).
    The tokenizer warns when a single encode() call exceeds that limit; we
    temporarily raise the limit so all tokens are kept for sliding windows.
    """
    old_max = tokenizer.model_max_length
    try:
        tokenizer.model_max_length = 10_000_000
        return tokenizer.encode(text, add_special_tokens=False)
    finally:
        tokenizer.model_max_length = old_max


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """
    Builds token windows across all pages, preserving page attribution per chunk.
    Empty pages are skipped; returns [] if no text is found.
    """
    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    tokenizer = _get_tokenizer()
    stride = size - overlap

    token_ids: list[int] = []
    page_for_token: list[int] = []

    for page_num, text in pages:
        if not text:
            continue
        ids = _encode_full_text(tokenizer, text)
        token_ids.extend(ids)
        page_for_token.extend([page_num] * len(ids))

    if not token_ids:
        return []

    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0

    while start < len(token_ids):
        end = min(start + size, len(token_ids))
        window_ids = token_ids[start:end]
        window_pages = page_for_token[start:end]

        content = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
        if content:
            page = _majority_page(window_pages)
            chunks.append(
                TextChunk(content=content, page=page, chunk_index=chunk_index)
            )
            chunk_index += 1

        if end >= len(token_ids):
            break
        start += stride

    return chunks


def _majority_page(pages: list[int]) -> int | None:
    if not pages:
        return None
    counts: dict[int, int] = {}
    for p in pages:
        counts[p] = counts.get(p, 0) + 1
    # Tie-break by lowest page number for deterministic results
    return max(counts, key=lambda k: (counts[k], -k))
