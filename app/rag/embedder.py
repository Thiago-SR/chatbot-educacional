from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.config import settings


_embedder_instance: "Embedder" | None = None


class Embedder:
    """Classe wrapper para gerar embeddings textuais com sentence-transformers."""

    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def embed_text(self, text: str) -> list[float]:
        """Gera embedding para um único texto e retorna o vetor normalizado."""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para um lote de textos e retorna vetores normalizados."""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


def get_embedder() -> Embedder:
    """Retorna a instância singleton do embedder."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
