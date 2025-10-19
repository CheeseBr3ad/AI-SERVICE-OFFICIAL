from pydantic_settings import BaseSettings
from pathlib import Path as PathLib


class Settings(BaseSettings):
    AI_API_KEY: str
    AI_API_URL: str
    AI_MODEL: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    EMBEDDING_MODEL: str  # "all-MiniLM-L6-v2"
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8010
    SERVER_RELOAD: bool = False
    GEMINI_API_KEY: str
    GEMINI_AI_MODEL: str  # gemini-2.5-flash gemini-2.0-flash-lite gemini-2.5-flash-lite

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instantiate settings and ensure directory exists
settings = Settings()


# Export uppercase variables for other modules to import
AI_API_KEY = settings.AI_API_KEY
AI_API_URL = settings.AI_API_URL
AI_MODEL = settings.AI_MODEL
QDRANT_URL = settings.QDRANT_URL
QDRANT_API_KEY = settings.QDRANT_API_KEY
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
SERVER_HOST = settings.SERVER_HOST
SERVER_PORT = settings.SERVER_PORT
SERVER_RELOAD = settings.SERVER_RELOAD
GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_AI_MODEL = settings.GEMINI_AI_MODEL

# Assume this config.py is in the 'config' folder
# Get the project root (parent of the folder containing this file)
PROJECT_ROOT = PathLib(__file__).resolve().parent.parent

# Define the uploads folder at the root
UPLOAD_DIR = PROJECT_ROOT / "uploads"

# Create the folder if it doesn't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

print(f"Uploads folder is at: {UPLOAD_DIR}")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".docx"}

QDRANT_DOCUMENT_COLLECTION_NAME = "documents"
QDRANT_MEETING_TRANSCRIPTS_COLLECTION_NAME = "meeting_transcripts"
