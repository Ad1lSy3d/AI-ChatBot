import os
from pathlib import Path
from dotenv import load_dotenv

# Load the .env file from the root directory
load_dotenv()

class Settings:
    PROJECT_NAME: str = "Bharti AXA RAG Chatbot Backend"
    
    # Path Resolution
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    VECTOR_STORE_DIR: str = str(BASE_DIR / "vector_store")
    
    # API Keys 
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

settings = Settings()