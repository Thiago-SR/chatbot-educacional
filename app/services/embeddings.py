from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import GOOGLE_API_KEY, DATA_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )


def load_documents(directory: Path = DATA_DIR) -> list:
    documents = []
    pdf_files = list(directory.glob("*.pdf"))

    if not pdf_files:
        return documents

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    return documents


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_or_create_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    chroma_path = str(CHROMA_DIR)

    if CHROMA_DIR.exists():
        return Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings,
        )

    documents = load_documents()
    if not documents:
        return Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings,
        )

    chunks = split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path,
    )
    return vectorstore


def ingest_documents() -> int:
    """Carrega documentos da pasta data/documents/ e indexa no ChromaDB. Retorna a quantidade de chunks criados."""
    documents = load_documents()
    if not documents:
        return 0

    chunks = split_documents(documents)
    embeddings = get_embeddings()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return len(chunks)
