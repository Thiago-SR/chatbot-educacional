"""
FastAPI application entry point.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # startup
    yield
    # shutdown
    await engine.dispose()


app = FastAPI(
    title="Educational Chatbot — Afro-Brazilian Culture",
    description="RAG-powered chatbot API focused on Afro-Brazilian culture.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    return {
        "status": "ok",
        "project": "chatbot-educacional",
        "version": "0.1.0",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}
