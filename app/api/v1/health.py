"""
Health Check Endpoint
Provides system status including ChromaDB, Ollama, and collection info.
"""

import logging
from datetime import datetime

from fastapi import APIRouter

from app.schemas.models import HealthResponse, ServiceStatus, CollectionInfo
from app.services.vector_store import vector_store_service
from app.services.ai_logic import ai_logic_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health Check",
    description="Cek status kesehatan seluruh komponen sistem.",
)
async def health_check() -> HealthResponse:
    """Check health of all system components."""

    # Check Vector Store (ChromaDB)
    vector_ok = vector_store_service.is_healthy()
    vector_status = ServiceStatus(
        status="healthy" if vector_ok else "unhealthy",
        detail="ChromaDB connected" if vector_ok else "ChromaDB unreachable",
    )

    # Check LLM (Ollama)
    llm_ok = ai_logic_service.is_healthy()
    llm_status = ServiceStatus(
        status="healthy" if llm_ok else "unhealthy",
        detail="Ollama LLM ready" if llm_ok else "Ollama LLM unreachable",
    )

    # Overall status
    all_healthy = vector_ok and llm_ok
    overall = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=overall,
        timestamp=datetime.now(),
        services={
            "vector_store": vector_status,
            "llm": llm_status,
        },
    )


@router.get(
    "/health/collection",
    response_model=CollectionInfo,
    summary="Collection Info",
    description="Informasi tentang koleksi dokumen di ChromaDB.",
)
async def collection_info() -> CollectionInfo:
    """Get information about the document collection."""
    info = vector_store_service.get_collection_info()
    return CollectionInfo(**info)
