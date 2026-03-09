# Chatbot Educacional - Cultura Afro-Brasileira

Um chatbot educacional interativo que utiliza RAG (Retrieval-Augmented Generation) para difundir e valorizar a cultura afro-brasileira. O projeto busca promover o conhecimento sobre história, tradições, religiões de matriz africana, música, culinária e contribuições da população negra para a formação da identidade brasileira.

## Stack

- **FastAPI** + Uvicorn
- **LangChain** como orquestrador RAG
- **Google Gemini** como LLM
- **ChromaDB** como banco de dados vetorial

## Como rodar

1. Crie e ative um ambiente virtual:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:

```bash
copy .env.example .env
```

Edite o arquivo `.env` e insira sua chave da API do Google (Gemini).

4. Adicione PDFs na pasta `data/documents/`.

5. Inicie o servidor:

```bash
uvicorn app.main:app --reload
```

6. Acesse a documentação interativa em: http://localhost:8000/docs
