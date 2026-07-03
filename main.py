import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

# Import the audio processing tools from your voice_service file
from voice_service import speech_to_text, text_to_speech

# Initialize your FastAPI web server application
app = FastAPI()

# This is a temporary query engine placeholder. 
# When you build your RAG pipeline in weeks 3-4, you will replace this function 
# with your actual database search and LLM response generation code.
def temporary_rag_query_engine(text_query: str) -> str:
    return f"Processed RAG context response for your query: {text_query}"


@app.get("/")
def read_root():
    """Simple health check endpoint to make sure your server is alive."""
    return {"status": "AI Chatbot Backend is running smoothly!"}


@app.post("/voice-query/")
async def full_voice_rag_endpoint(file: UploadFile = File(...)):
    """Accepts user audio voice query, runs it through RAG, and returns an audio file answer."""
    input_audio_path = f"input_{file.filename}"
    output_audio_path = "response_output.mp3"
    
    try:
        # 1. Save the incoming audio recording file from the client onto local disk
        with open(input_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Convert user speech into a plain text query string
        user_query_text = speech_to_text(input_audio_path)
        
        # 3. Pass text query to your generation pipeline
        ai_response_text = temporary_rag_query_engine(user_query_text)
        
        # 4. Convert AI text response back into spoken audio file (.mp3)
        text_to_speech(ai_response_text, output_audio_path)
        
        # 5. Clean up user audio upload file safely to keep disk pristine
        if os.path.exists(input_audio_path):
            os.remove(input_audio_path)
            
        # 6. Return response audio back to the user
        return FileResponse(
            path=output_audio_path, 
            media_type="audio/mpeg", 
            filename="response.mp3"
        )
        
    except Exception as e:
        return {"error": f"An error occurred processing the voice stream: {str(e)}"}