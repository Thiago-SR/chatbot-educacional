"""
FastAPI application entry point.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Educational Chatbot — Afro-Brazilian Culture",
    description="RAG-powered chatbot API focused on Afro-Brazilian culture.",
    version="0.1.0",
)


@app.get("/")
async def root() -> dict:
    return {
        "status": "ok",
        "project": "chatbot-educacional",
        "version": "0.1.0",
    }
