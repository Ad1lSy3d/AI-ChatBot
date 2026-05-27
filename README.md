# AI-ChatBot
A chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions from documents. The system supports OCR for scanned files and voice interaction using Speech-to-Text and Text-to-Speech technologies.

# Problem Statement

Organizations store large amounts of information in:
- PDFs
- Scanned documents
- SOPs
- Reports
- Policy documents
- Manuals

Traditional keyword-based search systems are often inefficient and fail to provide contextual answers.

This project aims to build an AI assistant capable of:
- Understanding enterprise documents
- Retrieving relevant information semantically
- Answering user queries contextually
- Supporting scanned documents through OCR
- Providing voice-based interaction

---

# Features

## Core Features
- Retrieval-Augmented Generation (RAG)
- Document Question Answering
- Semantic Search
- Context-Aware Responses
- Source-Based Retrieval

## OCR Features
- Scanned PDF support
- Image-to-text extraction
- OCR preprocessing pipeline

## Voice Features
- Speech-to-Text (STT)
- Text-to-Speech (TTS)
- Voice-based interaction

## Additional Features
- Multi-document querying
- Modular architecture
- Scalable backend APIs

# System Architecture

```text
                ┌────────────────────┐
                │    User Interface  │
                │ (Text / Voice UI) │
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
```

---

# Workflow / Pipeline

## Step-by-Step Workflow

1. User uploads a document or asks a query
2. OCR extracts text from scanned files/images
3. Extracted text is cleaned and preprocessed
4. Documents are split into chunks
5. Embeddings are generated for each chunk
6. Embeddings are stored in the vector database
7. Relevant chunks are retrieved using semantic similarity
8. Retrieved context is passed to the LLM
9. LLM generates a grounded response
10. Response is optionally converted into speech
