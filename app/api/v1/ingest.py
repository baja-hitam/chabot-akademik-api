"""
Document Ingestion Endpoint
Handles file upload and processing for the RAG knowledge base.
Supports PDF, Markdown, and TXT files.
"""

import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.schemas.models import CollectionInfo, DocumentCategory, IngestResponse
from app.services.vector_store import vector_store_service

logger = logging.getLogger(__name__)

settings = get_settings()

router = APIRouter(tags=["Ingestion"])

ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
MAX_FILE_SIZE_MB = 50


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload & Ingest Document",
    description=(
        "Upload dokumen akademik (PDF, Markdown, TXT) untuk diproses "
        "dan disimpan ke dalam knowledge base chatbot."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="File dokumen yang akan diingest"),
    category: DocumentCategory = Form(
        default=DocumentCategory.LAINNYA,
        description="Kategori dokumen akademik",
    ),
) -> IngestResponse:
    """
    Upload and ingest an academic document into the vector store.

    Process:
    1. Validate file format and size
    2. Save file temporarily
    3. Load, split, embed, and store in ChromaDB
    4. Return ingestion statistics
    """
    start_time = time.time()

    # ── Validate file extension ───────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nama file tidak valid.")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Format file '{file_ext}' tidak didukung. "
                f"Format yang didukung: {', '.join(ALLOWED_EXTENSIONS)}"
            ),
        )

    # ── Read and validate file size ───────────────────────────────
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Ukuran file ({size_mb:.1f}MB) melebihi batas {MAX_FILE_SIZE_MB}MB.",
        )

    # ── Save file to upload directory (non-blocking) ────────────────
    upload_path = settings.upload_path / file.filename
    await asyncio.to_thread(upload_path.write_bytes, content)
    logger.info("Saved uploaded file: %s (%.2f MB)", file.filename, size_mb)

    # ── Ingest the document (run in thread pool) ────────────────────
    # vector_store_service.ingest_document() is a blocking synchronous
    # function (PDF rendering, embedding generation, ChromaDB writes, and
    # OCR via Ollama are all CPU/IO-bound sync calls).  Running it directly
    # in this async endpoint would freeze the event loop for the entire
    # duration, preventing ALL other requests — including /chat — from
    # being served until ingestion completes.
    #
    # asyncio.to_thread() offloads the call to the default thread-pool
    # executor so the event loop stays free to handle concurrent requests.
    try:
        result = await asyncio.to_thread(
            vector_store_service.ingest_document,
            upload_path,
            category.value,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Ingestion failed for '%s': %s", file.filename, e)
        raise HTTPException(
            status_code=500,
            detail=f"Gagal memproses dokumen: {str(e)}",
        )

    processing_time = round(time.time() - start_time, 3)

    # Build a human-readable status message
    ocr_note = " (teks gambar diekstrak via doctr OCR)" if result["ocr_used"] else ""
    superseded_note = (
        f" Menggantikan {result['supersedes_count']} versi dokumen lama."
        if result["supersedes_count"]
        else ""
    )

    return IngestResponse(
        message=f"Dokumen '{file.filename}' berhasil diingest.{ocr_note}{superseded_note}",
        filename=file.filename,
        category=category.value,
        chunks_created=result["chunks_created"],
        processing_time=processing_time,
        document_year=result["document_year"],
        is_latest=True,
        supersedes_count=result["supersedes_count"],
        ocr_used=result["ocr_used"],
    )


@router.delete(
    "/ingest/{filename}",
    summary="Delete Document",
    description="Hapus semua chunk dari dokumen tertentu di knowledge base.",
)
async def delete_document(filename: str) -> dict:
    """Delete all chunks of a specific document from the vector store."""
    from app.repositories.chroma_repo import chroma_repo

    deleted_count = chroma_repo.delete_by_source(filename)
    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Dokumen '{filename}' tidak ditemukan.",
        )

    return {
        "message": f"Berhasil menghapus {deleted_count} chunk dari '{filename}'.",
        "deleted_chunks": deleted_count,
    }


@router.get(
    "/ingest/info",
    response_model=CollectionInfo,
    summary="Collection Info",
    description="Lihat informasi tentang knowledge base saat ini.",
)
async def get_collection_info() -> CollectionInfo:
    """Get current collection statistics."""
    info = vector_store_service.get_collection_info()
    return CollectionInfo(**info)
