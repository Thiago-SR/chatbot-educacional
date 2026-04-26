"""
SQLAlchemy models for the educational chatbot database.

Tables:
  - documents: metadata for each ingested PDF
  - chunks: text passages with their embedding vectors
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    """
    Represents a book/PDF that has been ingested into the system.
    Each document produces multiple chunks in the chunks table.
    """
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    filename: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Document filename={self.filename} chunks={self.total_chunks}>"


class Chunk(Base):
    """
    Represents a text passage extracted from a document.
    Stores the content and its embedding vector (384 dimensions).
    """
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 384-dimensional vector produced by paraphrase-multilingual-MiniLM-L12-v2
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)  # position within the document
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)   # source page number
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk doc={self.document_id} index={self.chunk_index}>"
