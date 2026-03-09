from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.config import GOOGLE_API_KEY
from app.services.embeddings import get_or_create_vectorstore

PROMPT_TEMPLATE = """Você é um assistente educacional especializado em cultura afro-brasileira.
Use os trechos de contexto abaixo para responder à pergunta do usuário.
Se você não souber a resposta com base no contexto fornecido, diga que não possui essa informação nos materiais disponíveis.
Responda sempre em português brasileiro de forma clara e didática.

Contexto:
{context}

Pergunta: {question}

Resposta:"""


def get_rag_chain() -> RetrievalQA:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )

    vectorstore = get_or_create_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


async def get_answer(question: str) -> dict:
    chain = get_rag_chain()
    result = chain.invoke({"query": question})

    sources = []
    for doc in result.get("source_documents", []):
        source = doc.metadata.get("source", "")
        if source and source not in sources:
            sources.append(source)

    return {
        "answer": result["result"],
        "sources": sources,
    }
