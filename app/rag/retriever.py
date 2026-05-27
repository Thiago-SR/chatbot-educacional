from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class Retriever:
    """Implementa a busca semântica no banco de dados usando pgvector."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Busca os chunks mais semelhantes ao embedding da query.

        Usa o operador pgvector <=> para ordenar por distância coseno.
        Retorna uma lista de dicionários com chunk_id, content, score, filename e chunk_index.
        """
        stmt = text(
            """
            SELECT
                c.id AS chunk_id,
                c.content,
                c.chunk_index,
                1 - (c.embedding <=> CAST(:query_vector AS vector)) AS score,
                d.filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE 1 - (c.embedding <=> CAST(:query_vector AS vector)) >= :threshold
            ORDER BY c.embedding <=> CAST(:query_vector AS vector)
            LIMIT :top_k
            """
        )
        query_vector_str = self._format_vector(query_embedding)
        result = await self.session.execute(
            stmt,
            {
                "query_vector": query_vector_str,
                "threshold": threshold,
                "top_k": top_k,
            },
        )
        rows = result.mappings().all()

        return [
            {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "score": float(row["score"]),
                "filename": row["filename"],
                "chunk_index": row["chunk_index"],
            }
            for row in rows
        ]

    @staticmethod
    def _format_vector(vector: list[float]) -> str:
        """Formata o vetor em string para um bind compatível com pgvector."""
        return "[" + ",".join(str(float(x)) for x in vector) + "]"
