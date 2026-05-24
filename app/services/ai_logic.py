import logging
import re
import time
from collections.abc import AsyncGenerator
from typing import Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.schemas.models import ChatResponse, SourceDocument
from app.services.vector_store import vector_store_service

logger = logging.getLogger(__name__)

settings = get_settings()

# ── System Prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = """Kamu adalah asisten akademik AI yang cerdas dan membantu untuk lingkungan kampus/universitas.

ATURAN UTAMA:
1. Jika informasi TIDAK ADA dalam konteks, jawab: "Maaf, saya tidak menemukan informasi tersebut dalam dokumen yang tersedia. Silakan hubungi bagian akademik untuk informasi lebih lanjut."
2. JANGAN mengarang atau menambah informasi di luar konteks.
3. Berikan jawaban yang jelas, terstruktur, dan mudah dipahami.

ATURAN DOKUMEN ([TERBARU] / [KEDALUWARSA]):
1. Selalu prioritaskan dokumen [TERBARU]. Jika ada perbedaan, gunakan yang [TERBARU].
2. Jika terpaksa menggunakan dokumen [KEDALUWARSA], beri tahu pengguna bahwa informasi mungkin sudah tidak berlaku dan sarankan konfirmasi ke akademik.
3. JANGAN sebutkan label ([TERBARU]/[KEDALUWARSA]), nama file, metadata, atau sumber dokumen dalam jawaban. Fokus hanya pada isi informasi.

KONTEKS DOKUMEN:
{context}
"""

USER_PROMPT = """Pertanyaan: {question}

Jawab pertanyaan di atas berdasarkan konteks dokumen yang telah diberikan."""


class AILogicService:
    """RAG pipeline service using LangChain + Ollama or Groq."""

    def __init__(self) -> None:
        self._llm: Union[ChatOllama, ChatGroq, None] = None
        self._chain = None

    # ── LLM Initialization ────────────────────────────────────────

    def _get_llm(self) -> Union[ChatOllama, ChatGroq]:
        """
        Returns:
            ChatOllama if LLM_PROVIDER="local", ChatGroq if LLM_PROVIDER="groq"
        """
        if self._llm is None:
            provider = settings.LLM_PROVIDER.lower()
            
            if provider == "groq":
                logger.info(
                    "Initializing Groq LLM: %s (API Key configured: %s)",
                    settings.GROQ_MODEL_NAME,
                    bool(settings.GROQ_API_KEY),
                )
                self._llm = ChatGroq(
                    model=settings.GROQ_MODEL_NAME,
                    api_key=settings.GROQ_API_KEY,
                    temperature=0.3,
                    max_tokens=1024,
                )
            else:  # default to local (Ollama)
                logger.info(
                    "Initializing Ollama LLM: %s at %s",
                    settings.OLLAMA_MODEL_NAME,
                    settings.OLLAMA_BASE_URL,
                )
                self._llm = ChatOllama(
                    model=settings.OLLAMA_MODEL_NAME,
                    base_url=settings.OLLAMA_BASE_URL,
                    temperature=0.3,
                    num_predict=1024,
                )
        return self._llm

    def _build_chain(self):
        """Build the LangChain RAG chain."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT),
            ]
        )

        llm = self._get_llm()
        chain = prompt | llm | StrOutputParser()
        return chain

    @staticmethod
    def _sanitize_answer_text(answer: str) -> str:
        """Remove document-source style references from model output."""
        if not answer:
            return answer

        cleaned = answer
        # Remove common citation-like phrases such as: "(Dokumen 3, Bab II, 2.1)"
        cleaned = re.sub(
            r"\(\s*dokumen\s+\d+[^)]*\)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Remove inline references: "Dokumen 3" / "dokumen 2"
        cleaned = re.sub(r"\bdokumen\s+\d+\b", "", cleaned, flags=re.IGNORECASE)
        # Remove explicit source labels if the model emits them
        cleaned = re.sub(
            r"\bsumber\s*:[^\n.]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Remove version labels if they leak into final answer
        cleaned = re.sub(r"\[(TERBARU|KEDALUWARSA)\]", "", cleaned)

        # Normalize whitespace after removals
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    # ── RAG: Full Response ────────────────────────────────────────

    async def get_answer(
        self,
        question: str,
        category: str | None = None,
        chat_history: list = None,
    ) -> ChatResponse:

        start_time = time.time()

        # 1. Retrieve relevant document chunks
        retrieved_docs = vector_store_service.search_similar(
            query=question,
            category=category,
        )

        # 2. Build context string with versioning annotations
        if not retrieved_docs:
            context = "Tidak ada dokumen yang ditemukan terkait pertanyaan ini."
        else:
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                version_tag = (
                    "[TERBARU]" if doc.get("is_latest", True) else "[KEDALUWARSA]"
                )
                year_info = (
                    f" Tahun: {doc['document_year']}"
                    if doc.get("document_year")
                    else ""
                )
                header = f"[{version_tag}{year_info}]"
                context_parts.append(f"{header}\n{doc['content']}")
            context = "\n\n---\n\n".join(context_parts)

        # 3. Generate answer with LLM
        chain = self._build_chain()
        raw_answer = await chain.ainvoke(
            {
                "context": context,
                "question": question
            }
        )
        # answer = self._sanitize_answer_text(raw_answer)
        answer = raw_answer

        # 4. Build source documents
        # sources = [
        #     SourceDocument(
        #         content=doc["content"][:300] + "..."
        #         if len(doc["content"]) > 300
        #         else doc["content"],
        #         source=doc["source"],
        #         category=doc["category"],
        #         relevance_score=doc["relevance_score"],
        #         document_year=doc.get("document_year"),
        #         is_latest=doc.get("is_latest", True),
        #         ocr_used=doc.get("ocr_used", False),
        #     )
        #     for doc in retrieved_docs
        # ]

        processing_time = round(time.time() - start_time, 3)

        # logger.info(
        #     "Answered question in %.3fs with %d sources",
        #     processing_time,
        #     len(sources),
        # )

        return ChatResponse(
            answer=answer,
            processing_time=processing_time,
        )

    # ── RAG: Streaming Response ───────────────────────────────────

    async def stream_answer(
        self,
        question: str,
        category: str | None = None,
        chat_history: list = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the RAG answer token by token.

        Yields individual tokens as they are generated for real-time
        UX via Server-Sent Events (SSE).

        Args:
            question: User's academic question.
            category: Optional category filter.

        Yields:
            Individual tokens/chunks of the answer.
        """
        # 1. Retrieve relevant document chunks
        retrieved_docs = vector_store_service.search_similar(
            query=question,
            category=category,
        )

        # 2. Build context with versioning annotations
        if not retrieved_docs:
            context = "Tidak ada dokumen yang ditemukan terkait pertanyaan ini."
        else:
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                version_tag = (
                    "[TERBARU]" if doc.get("is_latest", True) else "[KEDALUWARSA]"
                )
                year_info = (
                    f" Tahun: {doc['document_year']}"
                    if doc.get("document_year")
                    else ""
                )
                header = f"[{version_tag}{year_info}]"
                context_parts.append(f"{header}\n{doc['content']}")
            context = "\n\n---\n\n".join(context_parts)

        # 3. Stream LLM response
        chain = self._build_chain()

        async for chunk in chain.astream(
            {
                "context": context,
                "question": question,
                "chat_history": chat_history or [],
            }
        ):
            yield self._sanitize_answer_text(chunk)

    # ── Health Check ──────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """Check if the Ollama LLM is accessible."""
        try:
            llm = self._get_llm()
            # Simple ping - attempt to access the model config
            return llm.model is not None
        except Exception as e:
            logger.error("LLM health check failed: %s", e)
            return False


# ── Singleton instance ────────────────────────────────────────────
ai_logic_service = AILogicService()
