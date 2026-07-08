import re
import traceback  # Imported to expose the deep stack trace details
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from backend.services.rag_service import rag_engine
from backend.services.voice_service import voice_engine
from backend.services.stt_service import stt_engine  
from backend.services.cache_service import semantic_cache  

router = APIRouter(prefix="/api", tags=["Chat"])

# =====================================================================
# AUDIO CLEANING HELPER
# =====================================================================
def clean_text_for_speech(text: str) -> str:
    """Removes markdown formatting syntax so the TTS engine doesn't read it literally."""
    text = re.sub(r'[\*_]', '', text)
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =====================================================================
# DATA SCHEMAS
# =====================================================================
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    audio_base64: str

# =====================================================================
# TEXT GATEWAY ENDPOINT
# =====================================================================
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        query_vector = rag_engine.embeddings.embed_query(payload.message)
        
        cached_hit = semantic_cache.get_cached_response(payload.message, query_vector)
        if cached_hit:
            print("🚀 [REDIS CACHE HIT - TEXT] Serving instant response from memory!")
            return ChatResponse(
                query=payload.message,
                answer=cached_hit["answer"],
                sources=cached_hit["sources"],
                audio_base64=cached_hit["audio_base64"]
            )
            
        print("🔍 [REDIS CACHE MISS - TEXT] Running local FAISS extraction and Cloud LLM synthesis...")
        result = rag_engine.answer_query(payload.message, payload.session_id)
        
        speech_text = clean_text_for_speech(result["answer"])
        audio_payload = await voice_engine.text_to_speech_base64(speech_text)
        
        semantic_cache.set_cache_response(
            query_vector=query_vector,
            answer_text=result["answer"],
            sources=result["sources"],
            audio_base64=audio_payload
        )
        
        return ChatResponse(
            query=payload.message,
            answer=result["answer"],  
            sources=result["sources"],
            audio_base64=audio_payload
        )
    except Exception as e:
        print(f"TEXT ENDPOINT CRASH: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Text Error: {str(e)}")


# =====================================================================
# NEW: SESSION HISTORY RETRIEVAL ENDPOINT
# =====================================================================
@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Fetches full text conversation log arrays out of Redis to rebuild UI chat bubbles."""
    try:
        raw_history = rag_engine.get_session_history(session_id)
        
        # Re-structure data into a clean dict format standard frontend UI templates love
        formatted_messages = []
        for role, text in raw_history:
            # Map langchain internal tracking tags to classic frontend visual terms
            ui_role = "user" if role == "human" else "assistant"
            formatted_messages.append({"role": ui_role, "content": text})
            
        return {"session_id": session_id, "messages": formatted_messages}
    except Exception as e:
        print(f"HISTORY ENDPOINT CRASH: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to look up past logs: {str(e)}")


# =====================================================================
# SPEECH-IN / SPEECH-OUT VOICE GATEWAY ENDPOINT
# =====================================================================
@router.post("/chat/voice", response_model=ChatResponse)
async def voice_chat_endpoint(
    file: UploadFile = File(...), 
    session_id: str = Form("default")
):
    try:
        audio_data = await file.read()
        user_transcription = stt_engine.transcribe_audio_bytes(audio_data)
        
        if not user_transcription:
            raise HTTPException(status_code=400, detail="Could not understand audio input.")
            
        query_vector = rag_engine.embeddings.embed_query(user_transcription)
        
        cached_hit = semantic_cache.get_cached_response(user_transcription, query_vector)
        if cached_hit:
            print("[REDIS CACHE HIT - VOICE] Intercepted transcribed text and bypassed pipeline!")
            return ChatResponse(
                query=user_transcription,
                answer=cached_hit["answer"],
                sources=cached_hit["sources"],
                audio_base64=cached_hit["audio_base64"]
            )
            
        print("🔍 [REDIS CACHE MISS - VOICE] Running full local RAG pipeline...")
        rag_result = rag_engine.answer_query(user_transcription, session_id)
        
        speech_text = clean_text_for_speech(rag_result["answer"])
        audio_payload = await voice_engine.text_to_speech_base64(speech_text)
        
        semantic_cache.set_cache_response(
            query_vector=query_vector,
            answer_text=rag_result["answer"],
            sources=rag_result["sources"],
            audio_base64=audio_payload
        )
        
        return ChatResponse(
            query=user_transcription,
            answer=rag_result["answer"],
            sources=rag_result["sources"],
            audio_base64=audio_payload
        )
    except Exception as e:
        print(f"VOICE ENDPOINT CRASH: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Voice Loop Error: {str(e)}")