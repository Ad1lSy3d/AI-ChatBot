from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.endpoints import router as api_router

app = FastAPI(
    title="Bharti AXA RAG Chatbot Backend",
    description="Headless API engine powering our AI-RAG chatbot with voice support",
    version="0.1.0"
)

# Enforce secure CORS guidelines to accept incoming traffic from hosted Lovable.ai workspaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin for development testing
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Include our functional API route blueprint
app.include_router(api_router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Bharti AXA Life RAG Backend is running smoothly",
        "docs_url": "/docs"
    }