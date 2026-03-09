from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat

app = FastAPI(
    title="Chatbot Educacional - Cultura Afro-Brasileira",
    description="API de um chatbot educacional com RAG sobre cultura afro-brasileira.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/")
async def root():
    return {"message": "Chatbot Educacional - Cultura Afro-Brasileira"}
