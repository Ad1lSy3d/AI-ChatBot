from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from backend.services.rag_service import rag_engine
from backend.services.voice_service import voice_engine
from backend.services.stt_service import stt_engine  # Import our new ears

router = APIRouter(prefix="/api", tags=["Chat"])

# Text Schemas
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    audio_base64: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        result = rag_engine.answer_query(payload.message, payload.session_id)
        audio_payload = await voice_engine.text_to_speech_base64(result["answer"])
        return ChatResponse(
            query=payload.message,
            answer=result["answer"],
            sources=result["sources"],
            audio_base64=audio_payload
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Text Error: {str(e)}")


# =====================================================================
# NEW: SPEECH-IN / SPEECH-OUT VOICE GATEWAY ENDPOINT
# =====================================================================
@router.post("/chat/voice", response_model=ChatResponse)
async def voice_chat_endpoint(
    file: UploadFile = File(...), 
    session_id: str = Form("default")
):
    """Accepts a recorded audio file, transcribes it, runs RAG, and responds with Text + Voice."""
    try:
        # 1. Read binary audio files payload streaming from frontend microphone
        audio_data = await file.read()
        
        # 2. Local Faster-Whisper decoding: Speech ➔ Text String
        user_transcription = stt_engine.transcribe_audio_bytes(audio_data)
        
        if not user_transcription:
            raise HTTPException(status_code=400, detail="Could not understand audio input.")
            
        # 3. Feed the transcribed text directly into our established FAISS RAG loop
        rag_result = rag_engine.answer_query(user_transcription, session_id)
        
        # 4. Generate the Indian English response audio stream
        audio_payload = await voice_engine.text_to_speech_base64(rag_result["answer"])
        
        # 5. Return everything in one unified package
        return ChatResponse(
            query=user_transcription, # Let frontend see exactly what it heard you say!
            answer=rag_result["answer"],
            sources=rag_result["sources"],
            audio_base64=audio_payload
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Voice Loop Error: {str(e)}")