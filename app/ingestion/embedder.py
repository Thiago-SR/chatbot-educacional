"""
Local sentence-transformer embeddings (384 dimensions).
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

DEFAULT_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def encode_texts(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Returns one embedding vector per input string."""
    if not texts:
        return []

    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > batch_size,
        convert_to_numpy=True,
    )
    return [vec.tolist() for vec in vectors]
