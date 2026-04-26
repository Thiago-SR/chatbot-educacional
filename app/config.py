"""
Application settings loaded from environment variables.
Copy .env.example to .env and fill in the values.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://chatbot:chatbot123@localhost:5432/chatbot_edu"

    # Groq (LLM)
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Embeddings (local model, no API cost)
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # RAG parameters
    CHUNK_SIZE: int = 500       # chunk size in tokens
    CHUNK_OVERLAP: int = 50     # overlap between consecutive chunks
    TOP_K_RESULTS: int = 5      # number of chunks retrieved per query

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Global singleton
settings = Settings()
