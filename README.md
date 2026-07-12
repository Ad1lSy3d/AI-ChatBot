# Bharti AXA Assist: High-Availability Conversational RAG & Voice Engine

A multi-modal Retrieval-Augmented Generation (RAG) conversational AI assistant built for Bharti AXA Life Insurance. It enables users to ask policy-related questions through text or voice, retrieving grounded answers from internal insurance documents while minimizing hallucinations. The system combines semantic search, intelligent caching, conversational memory, and a multi-tier LLM failover mechanism to deliver fast and reliable responses.

---

# Features

- 📄 Retrieval-Augmented Generation (RAG)
- 🤖 3-Tier LLM Failover (Gemini → Groq → OpenRouter)
- ⚡ Redis Semantic Caching
- 💬 Persistent Conversation Memory
- 🎙️ Voice Support (Faster-Whisper + Edge-TTS)
- 📚 Context-aware document chunking

---

# Tech Stack

## Frontend
- React
- Vite
- Tailwind CSS
- Web Audio API

## Backend
- FastAPI
- Uvicorn
- LangChain

## Database / Storage
- Redis
- FAISS

## AI / LLMs
- Gemini 2.5 Flash
- Groq (Llama 3.1)
- OpenRouter
- Sentence Transformers
- Faster-Whisper
- Edge-TTS

## Deployment
- Vercel
- Local Backend
- ngrok

## Other Tools
- Astral uv
- PyPDF
- python-dotenv

---

# Architecture

The application follows a cache-first Retrieval-Augmented Generation (RAG) pipeline. Every request first checks the Redis semantic cache before executing the complete retrieval pipeline.

```text
+-----------------------+      ngrok HTTPS Tunnel      +-------------------------+
|  React Web UI Client  | <──────────────────────────> |   FastAPI REST Gateway  |
|  (Vercel Edge Stack)  |                              |    (Local Mac Node)     |
+-----------------------+                              +------------+------------+
                                                                     |
                                                                     v
                                                        +-------------------------+
                                                        |  Redis Semantic Cache   |
                                                        +------------+------------+
                                                                     |
                                          +──────────────────────────┴──────────────────────────+
                                          |                                                     |
                                 [🚀 CACHE HIT < 0.1]                                  [🔍 CACHE MISS >= 0.1]
                                          |                                                     |
                                          v                                                     v
                              +───────────────────────+                             +────────────────────────+
                              | Return Cached Payload |                             | Local Speech Decoder   |
                              | (Text, Audio, Source) |                             | (faster-whisper CPU)   |
                              +───────────────────────+                             +-----------+------------+
                                                                                                |
                                                                                                v
                                                                                    +────────────────────────+
                                                                                    |  FAISS Vector Search   |
                                                                                    |   (Context Chunks)     |
                                                                                    +-----------+------------+
                                                                                                |
                                                                                                v
                                                                                    +────────────────────────+
                                                                                    | Redis History Engine   |
                                                                                    |  (Session Tracking)    |
                                                                                    +-----------+------------+
                                                                                                |
                                                                                                v
                                                                                    +────────────────────────+
                                                                                    | 3-Tier LLM Failover    |
                                                                                    | Gemini → Groq → OR     |
                                                                                    +-----------+------------+
                                                                                                |
                                                                                                v
                                                                                    +────────────────────────+
                                                                                    | Local Voice Synthesis  |
                                                                                    | (edge-tts Base64 RAM)  |
                                                                                    +-----------+------------+
                                                                                                |
                                                                                                v
                                                                                    +────────────────────────+
                                                                                    | Cache Response (Redis) |
                                                                                    +-----------+------------+
                                                                                                |
                                                                                                v
                                                                                    +────────────────────────+
                                                                                    | JSON + Sources + Audio |
                                                                                    +────────────────────────+
```

---

# Project Structure

```text
backend/
├── api/
│   └── endpoints.py
├── core/
│   └── config.py
├── services/
│   ├── cache_service.py
│   ├── rag_service.py
│   ├── stt_service.py
│   └── voice_service.py
└── main.py

data/
└── raw/

scripts/
└── ingest.py

vector_store/

pyproject.toml
uv.lock
```

---

# Setup

## Prerequisites

- Python 3.11+
- Redis
- ngrok
- Astral uv

## Clone Repository

```bash
git clone https://github.com/your-username/AI-ChatBot.git
cd bharti-axa-assist-backend
```

## Install Dependencies

```bash
uv sync
```

## Configure `.env`

```env
PORT=8000
VECTOR_STORE_DIR=backend/vector_store
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
OPENROUTER_API_KEY=your_key
```

## Run Backend

```bash
brew services start redis
ngrok http 8000
uv run uvicorn backend.main:app --reload
```

## Run Frontend

Update the frontend API base URL with the generated ngrok HTTPS URL and start the frontend normally.

---

# Environment Variables

| Variable | Description |
|----------|-------------|
| PORT | Backend server port |
| VECTOR_STORE_DIR | FAISS vector store location |
| GEMINI_API_KEY | Gemini API key |
| GROQ_API_KEY | Groq API key |
| OPENROUTER_API_KEY | OpenRouter API key |

---

# API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Text chat |
| POST | `/api/chat/voice` | Voice chat |
| GET | `/api/chat/history/{session_id}` | Retrieve chat history |

---

# Deployment

**Frontend:** Vercel

**Backend:** Local FastAPI exposed through ngrok.

---

# Future Improvements

- Cloud vector database (Pinecone/Qdrant)
- Multilingual voice support
- Automatic document ingestion
- Docker deployment

---

# License

No license specified.
