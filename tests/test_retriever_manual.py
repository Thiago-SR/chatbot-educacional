"""Script de teste manual para validação do retriever semântico."""

from __future__ import annotations

import asyncio
from statistics import mean

from dotenv import load_dotenv

load_dotenv()  # Carrega variáveis do .env antes de importar app.config

from app.db.session import async_session_factory
from app.rag.embedder import Embedder, get_embedder
from app.rag.retriever import Retriever


QUESTOES = [
    "Quem foi Ibn Battuta e qual região africana ele visitou no século XIV?",
    "Qual era a importância do Deserto do Saara para as sociedades africanas antigas?",
    "Quem foi Sundjata Keita na história do Mali",
    "Qual era o objetivo do projeto Pós-Afrikas citado na apresentação do livro?",
]


def format_chunk(chunk: dict, index: int) -> str:
    return (
        f"{index}. score={chunk['score']:.4f} filename={chunk['filename']} "
        f"chunk_index={chunk['chunk_index']}\n"
        f"   {chunk['content'][:280].replace('\n', ' ')}"
    )


async def main() -> None:
    load_dotenv()  # Garante leitura das variáveis do .env
    embedder = get_embedder()

    async with async_session_factory() as session:
        retriever = Retriever(session)
        first_scores: list[float] = []

        for pergunta in QUESTOES:
            print(f"Pergunta: {pergunta}")
            query_embedding = embedder.embed_text(pergunta)
            chunks = await retriever.search(query_embedding, top_k=5, threshold=0.0)

            if chunks:
                first_scores.append(chunks[0]["score"])

            for index, chunk in enumerate(chunks, start=1):
                print(format_chunk(chunk, index))
                print()

            if not chunks:
                print("Nenhum chunk retornado para esta pergunta.")

            print("-" * 100)

        total = len(QUESTOES)
        average_first = mean(first_scores) if first_scores else 0.0

        print("Resumo:")
        print(f"  buscas realizadas: {total}")
        print(f"  score médio do primeiro resultado: {average_first:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
