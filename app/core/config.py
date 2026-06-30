"""
Application Configuration
Loads environment variables and provides centralized settings
for the Academic AI Chatbot Backend.
"""

from functools import lru_cache
from pathlib import Path
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── LLM Provider Configuration ────────────────────────────────
    # Choose 'local' for Ollama or 'groq' for Groq Cloud API
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "local")
    
    # ── Ollama Configuration (when LLM_PROVIDER="local") ─────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL")
    OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME")
    # Vision model used for image-based PDF OCR (must support image input).
    # Recommended options: gemma4:e4b, llava, moondream, llava-phi3
    # Set to empty string "" to disable Ollama vision OCR and fall back to doctr.
    OLLAMA_VISION_MODEL: str = os.getenv("OLLAMA_VISION_MODEL", "gemma4:e4b")
    
    # ── Groq Configuration (when LLM_PROVIDER="groq") ──────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768")
    
    # Backward compatibility
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "")
    
    # Render resolution for PDF pages sent to the doctr OCR pipeline.
    # 200 DPI is the recommended default: sharp enough for doctr to recognise
    # characters reliably while keeping PNG payloads reasonable (~3–5 MB).
    # Increase to 250–300 via the OCR_DPI env var only if text is very small
    # (e.g. dense tables with 6–8 pt fonts on scanned pages).
    OCR_DPI: int = int(os.getenv("OCR_DPI", 200))

    # ── Embedding Model ───────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"

    # ── ChromaDB Configuration ────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION_NAME: str = "academic_docs"

    # ── Document Processing ───────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── RAG Configuration ─────────────────────────────────────────
    USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
    TOP_K_RETRIEVE: int = int(os.getenv("TOP_K_RETRIEVE", 25))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 5))
    # Number of characters to keep as a short excerpt when returning
    # document snippets to the LLM. This helps reduce input token usage.
    EXCERPT_CHARS: int = 500

    # ── Database Configuration ────────────────────────────────────
    POSTGRES_URL: str = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/chatbot_akademik")
    
    # ── Security & JWT ────────────────────────────────────────────
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "super-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    # ── Application ───────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_DEBUG: bool = True

    # ── CORS ──────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8080"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def chroma_persist_path(self) -> Path:
        """Resolve the ChromaDB persistence directory as an absolute path."""
        return Path(self.CHROMA_PERSIST_DIR).resolve()

    # ── Upload Directory ──────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads"

    @property
    def upload_path(self) -> Path:
        """Resolve the upload directory as an absolute path."""
        path = Path(self.UPLOAD_DIR).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton for application settings."""
    return Settings()
