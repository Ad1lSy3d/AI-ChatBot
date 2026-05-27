# AI-ChatBot
A chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions from documents. The system supports OCR for scanned files and voice interaction using Speech-to-Text and Text-to-Speech technologies.

                ┌────────────────────┐
                │    User Interface  │
                │ (Text / Voice UI)  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   FastAPI Backend  │
                └─────────┬──────────┘
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
 ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
 │ Speech-to-Text │ │ OCR Pipeline │ │ Query Engine │
 │ Faster-Whisper │ │ PaddleOCR    │ │ RAG Pipeline │
 └────────────────┘ └──────┬───────┘ └──────┬───────┘
                            │                │
                            ▼                ▼
                    ┌────────────────────────────┐
                    │ Text Chunking & Embeddings │
                    └────────────┬───────────────┘
                                 ▼
                    ┌────────────────────────────┐
                    │  Vector Database           │
                    │  FAISS / ChromaDB          │
                    └────────────┬───────────────┘
                                 ▼
                    ┌────────────────────────────┐
                    │ Retriever + LLM            │
                    │ OpenAI / Ollama / Gemini   │
                    └────────────┬───────────────┘
                                 ▼
                    ┌────────────────────────────┐
                    │ Text-to-Speech             │
                    │ ElevenLabs / Coqui         │
                    └────────────────────────────┘
