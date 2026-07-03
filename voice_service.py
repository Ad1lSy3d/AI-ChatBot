import os
from faster_whisper import WhisperModel
from elevenlabs.client import ElevenLabs

# 1. Initialize local Speech-to-Text translation model
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# 2. Assign your valid secret API Key token string directly here
MY_SECRET_KEY = "sk_440fed87c12b1ab04d67a1b608228bdafb6ace95431e0c78"
tts_client = ElevenLabs(api_key=MY_SECRET_KEY)

def speech_to_text(audio_file_path: str) -> str:
    """Converts recorded user audio files into plaintext string inputs using Whisper."""
    segments, info = stt_model.transcribe(audio_file_path, beam_size=5)
    transcription = " ".join([segment.text for segment in segments])
    return transcription.strip()

def text_to_speech(text: str, output_audio_path: str):
    """Converts textual response strings into premium lifelike audio streams using Flash."""
    try:
        # Call the updated text_to_speech endpoint conversion layout
        response_stream = tts_client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",   # The premium "Adam" default profile ID
            model_id="eleven_flash_v2_5",       # Updated ultra-low latency production model name
            text=text
        )
        
        # Write incoming network sound blocks directly out to the target file path
        with open(output_audio_path, "wb") as f:
            for chunk in response_stream:
                if chunk:
                    f.write(chunk)
                    
    except Exception as e:
        print(f"ElevenLabs Processing Exception caught: {e}")
        # Build an empty file safe fallback so your app handles network issues cleanly
        with open(output_audio_path, "wb") as f:
            f.write(b"")