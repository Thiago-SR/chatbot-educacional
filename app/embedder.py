"""
Unified sentence-transformer embeddings (384 dimensions).

Single source of truth for embedding generation, used by both
the ingestion pipeline and the RAG query path.
All embeddings are normalized for cosine similarity.
"""

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model (thread-safe via lru_cache)."""
    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_text(text: str) -> list[float]:
    """Generate a normalized embedding for a single text."""
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def encode_texts(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Generate normalized embeddings for a batch of texts."""
    if not texts:
        return []

    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > batch_size,
        normalize_embeddings=True,
    )
    return [vec.tolist() for vec in vectors]
