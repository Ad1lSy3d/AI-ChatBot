import io
from faster_whisper import WhisperModel

class STTService:
    def __init__(self):
        print("Initializing Local Faster-Whisper Engine (Base Model)...")
        # 'device="cpu"' paired with 'compute_type="int8"' keeps it ultra-lightweight
        # and avoids complicated Apple Silicon compilation workarounds.
        self.model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Faster-Whisper initialized successfully.")

    def transcribe_audio_bytes(self, audio_bytes: bytes) -> str:
        """Takes raw in-memory audio binary data and returns transcribed text string."""
        try:
            # Wrap raw network bytes into an in-memory stream object
            audio_stream = io.BytesIO(audio_bytes)
            
            # FIXED: Added language="en" to enforce clean English character strings
            segments, info = self.model.transcribe(audio_stream, beam_size=5, language="en")
            
            # Combine individual audio segment text snippets cleanly
            transcribed_text = "".join([segment.text for segment in segments])
            return transcribed_text.strip()
            
        except Exception as e:
            print(f"Speech-to-Text Conversion Error: {str(e)}")
            return ""

# Initialize singleton listener instance
stt_engine = STTService()