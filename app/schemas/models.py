"""
Pydantic models for request/response validation.
Defines the data contracts for all API endpoints.
"""

from datetime import datetime
from enum import Enum
import uuid

from pydantic import BaseModel, Field

# ── Enums ─────────────────────────────────────────────────────────


class DocumentCategory(str, Enum):
    """Categories for academic documents."""

    KURIKULUM = "kurikulum"
    PERATURAN = "peraturan"
    PEDOMANPENULISANKKP = "pedoman_penulisan_kkp"
    PANDUANTOPIKKKP = "panduan_topik_kkp"
    PANDUANTOPIKTA = "panduan_topik_ta"
    JADWAL = "jadwal"
    INFORMASI_UMUM = "informasi_umum"
    BEASISWA = "beasiswa"
    AKADEMIK = "akademik"
    KEMAHASISWAAN = "kemahasiswaan"
    LAINNYA = "lainnya"


# ── Chat Schemas ──────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Pertanyaan pengguna tentang informasi akademik",
        examples=["Bagaimana cara mengajukan cuti akademik?"],
    )
    category: DocumentCategory | None = Field(
        default=None,
        description="Filter pencarian berdasarkan kategori dokumen (opsional)",
    )
    session_id: str | None = Field(
        default=None,
        description="ID Sesi chat (opsional, jika kosong akan membuat sesi baru)",
    )


class SourceDocument(BaseModel):
    content: str = Field(description="Potongan teks dari dokumen sumber")
    source: str = Field(description="Nama file atau sumber dokumen")
    category: str = Field(default="", description="Kategori dokumen")
    relevance_score: float = Field(
        default=0.0,
        description="Skor relevansi (0-1, semakin tinggi semakin relevan)",
    )
    document_year: int | None = Field(default=None, description="Tahun dokumen")
    is_latest: bool | None = Field(
        default=True, description="Apakah dokumen merupakan versi terbaru"
    )
    ocr_used: bool | None = Field(
        default=False, description="Apakah OCR digunakan untuk ekstraksi teks"
    )


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""

    answer: str = Field(description="Jawaban dari AI berdasarkan konteks dokumen")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Daftar dokumen sumber yang digunakan",
    )
    processing_time: float = Field(
        default=0.0,
        description="Waktu pemrosesan dalam detik",
    )
    session_id: str | None = Field(
        default=None,
        description="ID Sesi chat (opsional)",
    )


class ChatSessionResponse(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatMessageDetail(BaseModel):
    id: uuid.UUID
    sender_role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True

# ── Ingest Schemas ────────────────────────────────────────────────


class IngestResponse(BaseModel):
    """Response model for the document ingestion endpoint."""

    message: str = Field(description="Status pesan ingesti")
    filename: str = Field(description="Nama file yang diproses")
    category: str = Field(description="Kategori dokumen")
    chunks_created: int = Field(
        default=0,
        description="Jumlah chunk yang berhasil dibuat",
    )
    processing_time: float = Field(
        default=0.0,
        description="Waktu pemrosesan dalam detik",
    )
    document_year: int | None = Field(default=None, description="Tahun dokumen")
    is_latest: bool = Field(
        default=True, description="Apakah dokumen merupakan versi terbaru"
    )
    supersedes_count: int = Field(
        default=0,
        description="Jumlah versi lama yang ditandai sebagai superseded",
    )
    ocr_used: bool = Field(
        default=False, description="Apakah OCR digunakan untuk ekstraksi teks"
    )


# ── Health Schemas ────────────────────────────────────────────────


class ServiceStatus(BaseModel):
    """Status of an individual service component."""

    status: str = Field(description="Status komponen: 'healthy' atau 'unhealthy'")
    detail: str = Field(default="", description="Detail tambahan")


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str = Field(description="Status keseluruhan sistem")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Waktu pengecekan",
    )
    services: dict[str, ServiceStatus] = Field(
        default_factory=dict,
        description="Status masing-masing komponen",
    )


# ── Collection Info ───────────────────────────────────────────────


class CollectionInfo(BaseModel):
    """Information about the ChromaDB collection."""

    collection_name: str
    document_count: int
    categories: list[str] = Field(default_factory=list)
