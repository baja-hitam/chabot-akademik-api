"""
Vector Store Service
Handles document loading, text splitting, embedding generation,
and interaction with the ChromaDB repository.

Features:
- Vision-OCR fallback for image-based PDFs using DeepSeek OCR via Ollama
  (requires pdf2image; the OCR model must be available in your Ollama instance)
- Automatic document versioning: older versions of the same document are
  marked as superseded when a newer year is uploaded.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.repositories.chroma_repo import chroma_repo

logger = logging.getLogger(__name__)

settings = get_settings()

# Pages whose native-extracted text is shorter than this (chars) are
# considered image-only → OCR replaces the native text.
_OCR_THRESHOLD_CHARS_PER_PAGE = 200

# If the total bounding-box area of embedded images on a page exceeds
# this fraction of the page area, the page is flagged as containing
# significant visual content.
#
# Rationale for 0.25 (25 %):
#   - Typical header/footer decorations (logo + rule lines) cover ~18 % → excluded.
#   - Actual content images (screenshots, diagrams) start at ~31 % → included.
#   - A clean gap between 18 % and 31 % makes 25 % a stable threshold.
_OCR_IMAGE_AREA_RATIO = 0.25


class VectorStoreService:
    """Service for document processing, embedding, and vector storage."""

    def __init__(self) -> None:
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._ocr_predictor: Any = None  # lazy-init doctr OCR predictor
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Embedding Model ───────────────────────────────────────────

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-initialize the HuggingFace embedding model."""
        if self._embeddings is None:
            logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL_NAME)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded successfully")
        return self._embeddings

    # ── Document Loading ──────────────────────────────────────────

    def _load_file(self, file_path: Path) -> tuple[str, bool]:
        """
        Load text content from a file (PDF, Markdown, or TXT).

        Args:
            file_path: Path to the document file.

        Returns:
            Tuple of (extracted_text, ocr_was_used).

        Raises:
            ValueError: If the file format is unsupported.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix in (".md", ".markdown"):
            return self._load_text(file_path), False
        elif suffix == ".txt":
            return self._load_text(file_path), False
        else:
            raise ValueError(
                f"Format file tidak didukung: {suffix}. "
                "Gunakan PDF, Markdown (.md), atau TXT."
            )

    def _load_pdf(self, file_path: Path) -> tuple[str, bool]:
        """
        Extract text from a PDF with per-page, image-aware OCR.

        Each page is categorised into one of three buckets:

        ``text_only``
            No significant embedded images (image area < 25 % of page).
            Native pypdf extraction is used as-is.

        ``replace_ocr``
            Sparse native text (< ``_OCR_THRESHOLD_CHARS_PER_PAGE`` chars)
            **and** significant images.  The page is fully image-based;
            OCR result replaces the native text when it is longer.

        ``merge_ocr``
            Rich native text (>= threshold) **and** significant images.
            The page has both readable text and visual content (e.g. a
            screenshot inside a paragraph).  OCR result is appended after
            the native text so no information is lost.

        The OCR model is loaded once before processing and unloaded
        immediately afterwards via a ``finally`` block in
        :meth:`_load_pdf_with_ocr`.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Tuple of (extracted_text, ocr_was_used).
        """
        import fitz  # type: ignore[import-untyped]  # PyMuPDF
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)

        # ── Pass 1: classify every page ───────────────────────────────
        pages_text: list[str] = []
        replace_ocr: list[int] = []  # sparse text + significant images
        merge_ocr: list[int] = []  # rich text  + significant images

        fitz_doc = fitz.open(str(file_path))
        try:
            for i, pypdf_page in enumerate(reader.pages):
                fitz_page = fitz_doc[i]
                # ── Native text
                text = (pypdf_page.extract_text() or "").strip()
                pages_text.append(text)

                # ── Image-area ratio
                page_area = fitz_page.rect.width * fitz_page.rect.height
                if page_area > 0:
                    img_area = sum(
                        (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
                        for b in fitz_page.get_image_info()
                        if "bbox" in b
                    )
                    img_ratio = img_area / page_area
                else:
                    img_ratio = 0.0

                has_significant_image = img_ratio >= _OCR_IMAGE_AREA_RATIO
                is_sparse = len(text) < _OCR_THRESHOLD_CHARS_PER_PAGE

                if has_significant_image and is_sparse:
                    replace_ocr.append(i)
                elif has_significant_image and not is_sparse:
                    merge_ocr.append(i)
                # else: text_only — no OCR needed
        finally:
            fitz_doc.close()

        logger.info(
            "PDF page scan '%s': %d pages | replace-OCR: %s | merge-OCR: %s",
            file_path.name,
            total_pages,
            [p + 1 for p in replace_ocr],
            [p + 1 for p in merge_ocr],
        )

        # ── Pass 2: run OCR on flagged pages (model loaded/unloaded inside) ──
        ocr_used = False
        all_ocr_indices = sorted(set(replace_ocr + merge_ocr))

        if all_ocr_indices:
            ocr_map: dict[int, str] = dict(
                zip(
                    all_ocr_indices,
                    self._load_pdf_with_ocr(file_path, all_ocr_indices),
                )
            )

            for idx in replace_ocr:
                ocr_text = ocr_map.get(idx, "")
                if ocr_text and len(ocr_text) > len(pages_text[idx]):
                    pages_text[idx] = ocr_text
                    ocr_used = True

            for idx in merge_ocr:
                ocr_text = ocr_map.get(idx, "")
                if ocr_text:
                    # Append OCR result after native text with a clear separator
                    pages_text[idx] = (
                        pages_text[idx] + "\n\n[Konten gambar]\n" + ocr_text
                    )
                    ocr_used = True

        content = "\n\n".join(p for p in pages_text if p)
        logger.info(
            "PDF loaded '%s': %d total chars | OCR applied: %s",
            file_path.name,
            len(content),
            ocr_used,
        )
        return content, ocr_used

    # ── Local OCR (doctr) ─────────────────────────────────────

    def _get_ocr_predictor(self) -> Any:
        """
        Lazy-initialize the doctr OCR predictor.

        The predictor is created once per service instance and reused for all
        subsequent OCR calls.  On first use, doctr downloads two pre-trained
        models (~130 MB total) to ``~/.cache/doctr/models/`` and keeps them
        cached for future runs.
        """
        if self._ocr_predictor is None:
            from doctr.models import ocr_predictor  # type: ignore[import-untyped]

            logger.info(
                "Initializing doctr OCR predictor "
                "(first run downloads ~130 MB to ~/.cache/doctr/models/) …"
            )
            self._ocr_predictor = ocr_predictor(pretrained=True)
            logger.info("doctr OCR predictor ready.")
        return self._ocr_predictor

    def _load_pdf_with_ocr(
        self,
        file_path: Path,
        page_indices: list[int],
    ) -> list[str]:
        """
        OCR a specific subset of PDF pages using doctr (local, no network).

        Strategy
        --------
        Each requested page is rendered to a PNG image at ``settings.OCR_DPI``
        resolution by PyMuPDF, then passed to the doctr predictor which runs
        a two-stage pipeline:

        1. **Text detection** – FAST architecture (CNN) locates word bounding
           boxes on the image.
        2. **Text recognition** – CRNN model transcribes each detected region
           into a string.

        Both models run locally on CPU using PyTorch (already a project
        dependency via ``sentence-transformers``).  No Ollama, no network
        calls, no external binaries required.

        Args:
            file_path:    Path to the PDF file.
            page_indices: 0-based page numbers that need OCR.

        Returns:
            List of OCR'd text strings, one per entry in ``page_indices``.
            Failed pages return an empty string at that position.
        """
        try:
            import fitz  # type: ignore[import-untyped]  # PyMuPDF
            from doctr.io import DocumentFile  # type: ignore[import-untyped]
        except ImportError as exc:
            logger.warning(
                "OCR dependency missing (%s). "
                "Run: pip install pymupdf python-doctr  "
                "Skipping OCR for '%s'.",
                exc,
                file_path.name,
            )
            return [""] * len(page_indices)

        predictor = self._get_ocr_predictor()

        _dpi_scale = settings.OCR_DPI / 72
        _MATRIX = fitz.Matrix(_dpi_scale, _dpi_scale)
        logger.info(
            "doctr OCR: processing %d page(s) of '%s' at %d DPI…",
            len(page_indices),
            file_path.name,
            settings.OCR_DPI,
        )

        results: list[str] = []
        pdf_doc = None
        try:
            pdf_doc = fitz.open(str(file_path))
            for idx in page_indices:
                page_num = idx + 1
                try:
                    # Render page → PNG bytes (PyMuPDF, no external binary)
                    pix = pdf_doc[idx].get_pixmap(matrix=_MATRIX)
                    img_bytes = pix.tobytes("png")

                    # Run doctr OCR on this single page
                    doc_page = DocumentFile.from_images([img_bytes])
                    result = predictor(doc_page)

                    # Flatten blocks → lines → words into plain text
                    page_result = result.pages[0]
                    lines = [
                        " ".join(w.value for w in line.words)
                        for block in page_result.blocks
                        for line in block.lines
                    ]
                    text = "\n".join(line for line in lines if line.strip())
                    results.append(text)
                    logger.debug(
                        "doctr page %d of '%s': %d chars.",
                        page_num,
                        file_path.name,
                        len(text),
                    )

                except Exception as exc:
                    logger.error(
                        "doctr OCR failed on page %d of '%s': %s",
                        page_num,
                        file_path.name,
                        exc,
                    )
                    results.append("")

        finally:
            if pdf_doc is not None:
                try:
                    pdf_doc.close()
                except Exception:
                    pass

        logger.info(
            "doctr OCR completed for '%s': %d/%d pages extracted successfully.",
            file_path.name,
            sum(1 for r in results if r),
            len(page_indices),
        )
        return results

    def _load_text(self, file_path: Path) -> str:
        """Load plain text or Markdown file."""
        content = file_path.read_text(encoding="utf-8")
        logger.info("Loaded text file '%s': %d chars", file_path.name, len(content))
        return content

    # ── Text Splitting ────────────────────────────────────────────

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks using RecursiveCharacterTextSplitter."""
        chunks = self._text_splitter.split_text(text)
        logger.info("Split text into %d chunks", len(chunks))
        return chunks

    # ── Versioning Helpers ────────────────────────────────────────

    @staticmethod
    def _extract_year(filename: str) -> int | None:
        """
        Extract a 4-digit year (19xx or 20xx) from a filename.

        Takes the *last* year found so that names like
        'panduan_krs_2024_rev_2025.pdf' resolve to 2025.

        Args:
            filename: Filename string (with or without extension).

        Returns:
            Integer year, or None if no year is found.
        """
        # Use digit lookaround instead of \b because underscores are word chars
        matches = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", filename)
        return int(matches[-1]) if matches else None

    @staticmethod
    def _extract_base_name(filename: str) -> str:
        """
        Derive a normalized base name from a filename by removing the year
        token and file extension, then lowercasing and collapsing separators.

        Examples:
            'Panduan KRS 2025.pdf'  -> 'panduan_krs'
            'panduan-krs-2026.pdf'  -> 'panduan_krs'
            'kurikulum2024.pdf'     -> 'kurikulum'

        Args:
            filename: Original filename.

        Returns:
            Normalized base name string.
        """
        stem = Path(filename).stem  # Strip extension
        # Remove year tokens together with any surrounding separators
        base = re.sub(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", " ", stem)
        # Lowercase and collapse whitespace / punctuation into underscores
        base = re.sub(r"[\s_\-]+", "_", base.strip()).strip("_").lower()
        return base or stem.lower()

    def _manage_document_versions(
        self,
        new_source: str,
        new_base_name: str,
        new_year: int | None,
    ) -> int:
        """
        Mark older versions of the same document as superseded.

        An existing document is superseded when:
        - The new document has a year >= the existing one (or either has no year).

        This ensures there is always exactly one "latest" version per base name.

        Args:
            new_source:    Filename of the document being ingested.
            new_base_name: Normalized base name (e.g. 'panduan_krs').
            new_year:      Year extracted from the new filename, or None.

        Returns:
            Number of distinct source documents marked as superseded.
        """
        existing_sources = chroma_repo.get_sources_by_base_name(new_base_name)
        # Exclude the incoming source itself (handles re-ingestion gracefully)
        existing_sources = [s for s in existing_sources if s != new_source]

        superseded_count = 0
        for old_source in existing_sources:
            old_year = self._extract_year(old_source)

            if new_year is None:
                # No year info on new doc → conservatively supersede existing
                should_supersede = True
            elif old_year is None:
                # Old doc has no year but new one does → new is more explicit
                should_supersede = True
            else:
                should_supersede = new_year >= old_year

            if should_supersede:
                updated = chroma_repo.update_metadata_by_source(
                    old_source,
                    {"is_latest": False, "superseded_by": new_source},
                )
                if updated:
                    logger.info(
                        "Marked '%s' (year=%s) as superseded by '%s' (year=%s) — %d chunks updated.",
                        old_source,
                        old_year,
                        new_source,
                        new_year,
                        updated,
                    )
                    superseded_count += 1

        return superseded_count

    # ── Ingestion Pipeline ────────────────────────────────────────

    def ingest_document(
        self,
        file_path: Path,
        category: str = "lainnya",
    ) -> dict[str, Any]:
        """
        Full pipeline: load file → split → embed → store in ChromaDB.

        Handles OCR for image-based PDFs and manages document versioning
        so that older uploads of the same document are marked as superseded.

        Args:
            file_path: Path to the document file.
            category:  Document category for metadata filtering.

        Returns:
            Dict with keys:
                chunks_created   (int)      — number of chunks stored
                ocr_used         (bool)     — whether OCR was applied
                document_year    (int|None) — year extracted from filename
                supersedes_count (int)      — number of older versions superseded
        """
        # 1. Load document
        content, ocr_used = self._load_file(file_path)
        if not content.strip():
            raise ValueError(f"File '{file_path.name}' kosong atau tidak dapat dibaca.")

        # 2. Split into chunks
        chunks = self._split_text(content)
        if not chunks:
            raise ValueError("Tidak ada chunk yang dihasilkan dari dokumen.")

        # 3. Delete existing chunks for this source (re-ingestion support)
        chroma_repo.delete_by_source(file_path.name)

        # 4. Resolve versioning metadata
        document_year = self._extract_year(file_path.name)
        document_base_name = self._extract_base_name(file_path.name)
        supersedes_count = self._manage_document_versions(
            new_source=file_path.name,
            new_base_name=document_base_name,
            new_year=document_year,
        )

        # 5. Generate embeddings
        embeddings_model = self._get_embeddings()
        embeddings = embeddings_model.embed_documents(chunks)

        # 6. Prepare metadata and IDs
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(file_path.name, i, chunk)
            ids.append(chunk_id)
            # Create a short excerpt to reduce tokens when returning results
            raw_excerpt = re.sub(r"\s+", " ", chunk).strip()
            excerpt = raw_excerpt[: settings.EXCERPT_CHARS]

            metadatas.append(
                {
                    "source": file_path.name,
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    # Versioning fields
                    "document_year": document_year if document_year is not None else 0,
                    "document_base_name": document_base_name,
                    "is_latest": True,
                    # Extraction method flag
                    "ocr_used": ocr_used,
                    # Short excerpt for lightweight retrieval
                    "excerpt": excerpt,
                }
            )

        # 7. Store in ChromaDB
        chroma_repo.add_documents(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        logger.info(
            "Ingested '%s' [%s]: %d chunks | year=%s | base='%s' | ocr=%s | superseded=%d",
            file_path.name,
            category,
            len(chunks),
            document_year,
            document_base_name,
            ocr_used,
            supersedes_count,
        )

        return {
            "chunks_created": len(chunks),
            "ocr_used": ocr_used,
            "document_year": document_year,
            "supersedes_count": supersedes_count,
        }

    # ── Retrieval ─────────────────────────────────────────────────

    def search_similar(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for document chunks most similar to the query.

        Results include versioning metadata so the LLM prompt builder can
        signal to the model which documents are current and which are expired.

        Args:
            query:    The user's question.
            top_k:    Number of results to return.
            category: Optional category filter.

        Returns:
            List of dicts with keys: content, source, category,
            relevance_score, document_year, is_latest, ocr_used.
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS

        embeddings_model = self._get_embeddings()
        query_embedding = embeddings_model.embed_query(query)

        where_filter = None
        if category:
            where_filter = {"category": category}

        results = chroma_repo.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where_filter,
        )

        documents: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for i, _doc_id in enumerate(results["ids"][0]):
                distance = (
                    results["distances"][0][i] if results.get("distances") else 0.0
                )
                # Convert cosine distance to similarity score (range 0–1)
                score = max(0.0, 1.0 - distance / 2.0)

                # metadata: dict[str, Any] = (
                #     results["metadatas"][0][i] if results.get("metadatas") else {}
                # )
                # Prefer the precomputed excerpt in metadata to minimize tokens.
                metadata: dict[str, Any] = (
                    results["metadatas"][0][i] if results.get("metadatas") else {}
                )

                content: str = (
                    results["documents"][0][i] if results.get("documents") else ""
                )
                doc_year_raw = metadata.get("document_year", 0)
                document_year = int(doc_year_raw) if doc_year_raw else None

                documents.append(
                    {
                        "content": content,
                        "source": metadata.get("source", "unknown"),
                        "category": metadata.get("category", ""),
                        "relevance_score": round(score, 4),
                        "document_year": document_year,
                        "is_latest": bool(metadata.get("is_latest", True)),
                        "ocr_used": bool(metadata.get("ocr_used", False)),
                    }
                )

        logger.debug(
            "Search returned %d results for query: '%s'", len(documents), query[:80]
        )
        return documents

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _generate_chunk_id(source: str, index: int, content: str) -> str:
        """Generate a deterministic SHA-256-based ID for a chunk."""
        raw = f"{source}::{index}::{content[:100]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_collection_info(self) -> dict[str, Any]:
        """Proxy to the repository's collection info."""
        return chroma_repo.get_collection_info()

    def is_healthy(self) -> bool:
        """Check if the vector store is accessible."""
        try:
            chroma_repo.get_collection_info()
            return True
        except Exception as e:
            logger.error("Vector store health check failed: %s", e)
            return False


# ── Singleton instance ────────────────────────────────────────────
vector_store_service = VectorStoreService()
