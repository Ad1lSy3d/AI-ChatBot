import base64
import edge_tts

class VoiceService:
    def __init__(self):
        # Professional Indian English Neural Voice matching Bharti AXA's context
        self.default_voice = "en-IN-NeerjaNeural"

    async def text_to_speech_base64(self, text: str) -> str:
        """Converts text to speech entirely in memory and returns a Base64 string."""
        if not text.strip():
            return ""
            
        try:
            communicate = edge_tts.Communicate(text, self.default_voice)
            audio_bytes = b""
            
            # Stream the audio chunks natively into memory RAM without writing to disk
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            
            # Encode raw binary mp3 data into a clean text string for our JSON endpoint
            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            return base64_audio
            
        except Exception as e:
            print(f"⚠️ Voice Generation Error: {str(e)}")
            return ""

# Initialize singleton voice manager
voice_engine = VoiceService()