"""
Academic AI Chatbot Backend — Application Entry Point

FastAPI application with RAG-powered chatbot using:
- LangChain for orchestration
- ChromaDB for vector storage
- Gemma 4 E4B via Ollama for generation
- HuggingFace BGE-M3 for embeddings
"""

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request

from app.api.v1 import chat, health, ingest, auth, announcements
from app.core.config import get_settings
from app.infrastructure.database import engine
from app.domain.models import base  # this ensures models are registered

# Create database tables
base.Base.metadata.create_all(bind=engine)

# ── Logging Configuration ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Suppress noisy third-party HTTP client logs that expose internal
# Ollama requests (e.g. "POST http://localhost:11434/api/chat") and
# clutter the application log with implementation details.
for _noisy_logger in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Application Lifespan ──────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events for the application."""
    logger.info("=" * 60)
    logger.info("  Academic AI Chatbot Backend — Starting Up")
    logger.info("=" * 60)
    logger.info("  LLM Model   : %s", settings.LLM_MODEL_NAME)
    logger.info("  Ollama URL  : %s", settings.OLLAMA_BASE_URL)
    logger.info("  Embedding   : %s", settings.EMBEDDING_MODEL_NAME)
    logger.info("  ChromaDB    : %s", settings.chroma_persist_path)
    logger.info(
        "  Chunk Size  : %d (overlap: %d)", settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    )
    logger.info("  Top-K       : %d", settings.TOP_K_RESULTS)
    logger.info("=" * 60)

    yield

    logger.info("Academic AI Chatbot Backend — Shutting Down")


# ── FastAPI Application ───────────────────────────────────────────

app = FastAPI(
    title="Academic AI Chatbot API",
    description=(
        "API backend untuk chatbot akademik cerdas berbasis RAG "
        "(Retrieval-Augmented Generation). Menggunakan Llama 3.2 3B "
        "untuk menjawab pertanyaan berdasarkan dokumen akademik kampus."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "responseStatus": False,
            "responseMessage": exc.detail,  # Mengambil string pesan dari exc.detail
        }
    )


# ── CORS Middleware ───────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Register Routers ─────────────────────────────────────────────

app.include_router(health.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1/auth")
app.include_router(announcements.router, prefix="/api/v1/announcements")



# ── Root Endpoint ─────────────────────────────────────────────────


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Academic AI Chatbot API",
        "version": "1.0.0",
        "description": "RAG-powered chatbot untuk lingkungan akademik",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ── Run with Uvicorn ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_DEBUG,
    )
