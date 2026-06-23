from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.rag_service import rag_engine

router = APIRouter(prefix="/api", tags=["Chat"])

# Define the structure of the incoming request from Lovable
class ChatRequest(BaseModel):
    message: str

# Define the structure of the JSON response sent back to Lovable
class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
    try:
        # Pass user question into our RAG service
        result = rag_engine.answer_query(payload.message)
        
        return ChatResponse(
            query=payload.message,
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal AI Error: {str(e)}")