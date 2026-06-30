"""
Vector Store Service
Handles document loading, text splitting, embedding generation,
and interaction with the ChromaDB repository.

Features:
- Vision-OCR for image-based PDFs using Ollama multimodal models
  (e.g., gemma3, llava, moondream) as the primary OCR engine.
  These models understand Indonesian language context and handle
  scanned academic documents, tables, and schedules correctly.
- Fallback OCR via doctr (local, CPU) when Ollama Vision is unavailable.
- Automatic document versioning: older versions of the same document are
  marked as superseded when a newer year is uploaded.
"""

import hashlib
import logging
import re
import base64
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.repositories.chroma_repo import chroma_repo
from app.services.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

settings = get_settings()

# Pages whose native-extracted text is shorter than this (chars) are
# considered image-only → OCR replaces the native text.
#
# Rationale for 100 chars (down from 200):
#   - Scanned pages often yield a handful of noise characters from pypdf
#     (e.g. stray punctuation, header artefacts) that sum to 50–150 chars.
#   - Setting the bar at 100 ensures those noisy pages are still sent to OCR
#     while clean text-only pages (typically 500+ chars) are unaffected.
_OCR_THRESHOLD_CHARS_PER_PAGE = 100

# If the total bounding-box area of embedded images on a page exceeds
# this fraction of the page area, the page is flagged as containing
# significant visual content.
#
# Rationale for 0.15 (15 %):
#   - Tiny decorative elements (small logos, thin rule lines) cover < 10 % → excluded.
#   - Medium content images (screenshots, embedded figures) start at ~16 % → included.
#   - The 0.15 threshold is more sensitive than 0.25 and correctly captures
#     pages where a figure occupies only a quarter of the column width.
_OCR_IMAGE_AREA_RATIO = 0.15


class VectorStoreService:
    """Service for document processing, embedding, and vector storage."""

    def __init__(self) -> None:
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._reranker: CrossEncoder | None = None
        self._ocr_predictor: Any = None  # lazy-init doctr OCR predictor
        self._preprocessor = TextPreprocessor()
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

    def _get_reranker(self) -> CrossEncoder | None:
        """Lazy-initialize the CrossEncoder reranker model."""
        if not getattr(settings, "USE_RERANKER", False):
            return None
        if self._reranker is None:
            logger.info("Loading Reranker model: %s", settings.RERANKER_MODEL_NAME)
            self._reranker = CrossEncoder(settings.RERANKER_MODEL_NAME, max_length=512, device="cpu")
            logger.info("Reranker model loaded successfully")
        return self._reranker

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
                    self._load_pdf_with_doctr(file_path, all_ocr_indices),
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

    # ── Ollama Vision OCR ─────────────────────────────────────

    def _load_pdf_with_ollama_vision(
        self,
        file_path: Path,
        page_indices: list[int],
    ) -> list[str]:
        """
        Extract text from specific PDF pages using an Ollama multimodal model.

        This is the primary OCR strategy for image-heavy pages because
        multimodal LLMs (gemma3, llava, etc.) understand Indonesian language,
        academic document structure, tables, and scanned text far better than
        traditional OCR engines like doctr.

        Strategy
        --------
        Each requested page is rendered to a PNG image at ``settings.OCR_DPI``
        using PyMuPDF, then sent to the Ollama vision model with a prompt that
        instructs it to extract all visible text faithfully, preserving the
        original formatting (tables, bullet lists, headings) in plain text.

        Args:
            file_path:    Path to the PDF file.
            page_indices: 0-based page numbers that need OCR.

        Returns:
            List of extracted text strings, one per entry in ``page_indices``.
            Failed pages return an empty string at that position.
        """
        vision_model = getattr(settings, "OLLAMA_VISION_MODEL", "").strip()
        ollama_url = getattr(settings, "OLLAMA_BASE_URL", "").strip()

        if not vision_model or not ollama_url:
            logger.info(
                "Ollama Vision OCR skipped: OLLAMA_VISION_MODEL or OLLAMA_BASE_URL not set."
            )
            return [""] * len(page_indices)

        try:
            import ollama as ollama_client  # type: ignore[import-untyped]
            import fitz  # type: ignore[import-untyped]  # PyMuPDF
        except ImportError as exc:
            logger.warning(
                "Ollama Vision OCR dependency missing (%s). Skipping for '%s'.",
                exc,
                file_path.name,
            )
            return [""] * len(page_indices)

        # Use 96 DPI and JPEG for a compact payload (~150–300 KB per page).
        # The ollama library handles base64 encoding internally when bytes are passed.
        _VISION_DPI = 150
        _vis_scale = _VISION_DPI / 72
        _MATRIX = fitz.Matrix(_vis_scale, _vis_scale)

        _OCR_PROMPT = (
            "Kamu adalah mesin OCR yang sangat akurat untuk dokumen akademik berbahasa Indonesia. "
            "Tugas kamu adalah mengekstrak SEMUA teks yang terlihat pada gambar halaman dokumen ini "
            "secara lengkap dan akurat.\n\n"
            "Panduan ekstraksi:\n"
            "- Salin SEMUA teks persis seperti yang tertulis di dokumen, termasuk angka, tanggal, dan kode.\n"
            "- Pertahankan struktur tabel dalam format yang mudah dibaca (pisahkan kolom dengan ' | ').\n"
            "- Pertahankan judul, sub-judul, dan hierarki teks.\n"
            "- Jangan tambahkan penjelasan, komentar, atau interpretasi apapun.\n"
            "- Jika ada teks yang tidak jelas, tulis sesuai perkiraan terbaik kamu.\n"
            "- Mulai langsung dengan teks yang diekstrak, tanpa kata pengantar."
        )

        # Confusion markers: the model responded without seeing the image
        _CONFUSION_MARKERS = (
            "mohon berikan", "mohon lampirkan", "silakan berikan",
            "silakan kirim", "please provide", "please send", "please share",
        )

        # Instantiate Ollama client pointing at the configured base URL
        client = ollama_client.Client(host=ollama_url)

        logger.info(
            "Ollama Vision OCR: processing %d page(s) of '%s' using model '%s' at %s …",
            len(page_indices),
            file_path.name,
            vision_model,
            ollama_url,
        )

        results: list[str] = []
        pdf_doc = None
        try:
            pdf_doc = fitz.open(str(file_path))
            for idx in page_indices:
                page_num = idx + 1
                try:
                    # Render page → JPEG bytes (compact, enough detail for OCR)
                    pix = pdf_doc[idx].get_pixmap(matrix=_MATRIX)
                    img_bytes = pix.tobytes("jpeg")
                    # ── PERBAIKAN 3: Ubah ke Base64 String secara eksplisit untuk stabilitas Ollama ──
                    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    logger.debug(
                        "Ollama Vision: page %d rendered to JPEG, %d bytes.",
                        page_num, len(img_bytes),
                    )

                    # Use the official ollama client — it handles base64 encoding
                    # and the correct multimodal message format internally.
                    response = client.chat(
                        model=vision_model,
                        messages=[
                            {
                                "role": "user",
                                "content": _OCR_PROMPT,
                                "images": [img_base64],
                            }
                        ],
                        options={"temperature": 0, "num_predict": 4096},
                        stream=False
                    )
                    text = response.message.content.strip()

                    # Detect confusion responses (model didn't see the image)
                    if any(m in text.lower() for m in _CONFUSION_MARKERS):
                        logger.warning(
                            "Ollama Vision confusion response on page %d of '%s' — "
                            "model did not process the image. Falling back to doctr.",
                            page_num, file_path.name,
                        )
                        text = ""

                    results.append(text)
                    logger.info(
                        "Ollama Vision OCR page %d of '%s': %d chars.",
                        page_num,
                        file_path.name,
                        len(text),
                    )

                except Exception as exc:
                    logger.error(
                        "Ollama Vision OCR failed on page %d of '%s': %s",
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
            "Ollama Vision OCR completed for '%s': %d/%d pages extracted successfully.",
            file_path.name,
            sum(1 for r in results if r),
            len(page_indices),
        )
        return results

    # ── Local OCR (doctr — fallback) ────────────────────────────

    def _repair_text_with_ollama(self, raw_ocr_text: str) -> str:
        """
        Menggunakan LLM lokal untuk merapikan, memperbaiki typo, 
        dan merekonstruksi tabel dari teks mentah hasil OCR doctr.
        """
        # Pastikan self.settings diakses dengan benar lewat instance 'self'
        ollama_url = getattr(settings, "OLLAMA_BASE_URL", "").strip()
        repair_model = getattr(settings, "OLLAMA_REPAIR_MODEL", "gemma4:e4b").strip() 

        if not raw_ocr_text.strip():
            return ""

        if not ollama_url:
            logger.warning("OLLAMA_BASE_URL tidak diatur, mengembalikan teks mentah.")
            return raw_ocr_text

        # # ── PERBAIKAN 1: Pastikan endpoint mengarah ke /api/chat ──
        if not ollama_url.endswith("/api/chat"):
            ollama_url = f"{ollama_url.rstrip('/')}/api/chat"

        _REPAIR_PROMPT = (
            "Kamu adalah asisten ahli pemrosesan dokumen akademik Indonesia. "
            "Tugasmu adalah memperbaiki teks mentah hasil OCR yang berantakan di bawah ini agar menjadi rapi, "
            "layak dibaca, dan kaya akan kata kunci untuk sistem pencarian (knowledge base).\n\n"
            "Aturan Perbaikan:\n"
            "1. Perbaiki typo akibat salah baca OCR (contoh: '0' jadi 'O', '1' jadi 'l', kata hancur seperti 'Semsster').\n"
            "2. Jika teks tersebut terlihat seperti tabel/kalender kegiatan, susun ulang menjadi format tabel yang rapi menggunakan pemisah ' | '.\n"
            "3. Pertahankan semua data penting: tanggal, angka, tahun akademik, kode, dan nama kegiatan (JANGAN DIUBAH ATAU DIHAPUS).\n"
            "4. JANGAN tambahkan komentar, penjelasan, atau pengantar. Langsung keluarkan teks yang sudah diperbaiki."
        )

        try:
            import requests
            payload = {
                "model": repair_model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{_REPAIR_PROMPT}\n\nBerikut adalah teks mentah OCR:\n{raw_ocr_text}"
                    }
                ],
                "options": {
                    "temperature": 0.1
                },
                "stream": False
            }

            logger.info("Mengirim permintaan repair ke Ollama di %s menggunakan model %s...", ollama_url, repair_model)
            response = requests.post(ollama_url, json=payload, timeout=600)
            
            # ── PERBAIKAN 2: Tangani jika status code bukan 200 untuk mempermudah debugging ──
            if response.status_code != 200:
                logger.error("Ollama mengembalikan status code %d: %s", response.status_code, response.text)
                return raw_ocr_text

            # ── PERBAIKAN 3: Ambil response content secara aman ──
            response_data = response.json()
            repaired_text = response_data.get("message", {}).get("content", "").strip()
            
            if repaired_text:
                logger.info("Teks berhasil diperbaiki oleh Ollama.")
                print(repaired_text)  # Sekarang ini akan muncul jika respons berhasil
                return repaired_text
            else:
                logger.warning("Ollama berhasil merespons tetapi mengembalikan teks kosong.")
            
        except Exception as exc:
            logger.error("Gagal memperbaiki teks via Ollama: %s", exc)
        
        return raw_ocr_text

    def _get_ocr_predictor(self) -> Any:
        """
        Lazy-initialize the doctr OCR predictor (fallback OCR engine).

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

    def _load_pdf_with_doctr(
        self,
        file_path: Path,
        page_indices: list[int],
    ) -> list[str]:
        """
        OCR a specific subset of PDF pages using doctr (local fallback).

        Used when Ollama Vision is not available or returns empty results.
        Runs locally on CPU via PyTorch with no network calls required.

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
                "doctr OCR dependency missing (%s). "
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
            "doctr OCR (fallback): processing %d page(s) of '%s' at %d DPI…",
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

                    # Flatten blocks → lines → words into plain text.
                    # Sort blocks in reading order: top-to-bottom, then
                    # left-to-right within the same vertical band.  This is
                    # critical for multi-column layouts where doctr may return
                    # blocks in bounding-box order rather than reading order.
                    page_result = result.pages[0]
                    sorted_blocks = sorted(
                        page_result.blocks,
                        key=lambda b: (
                            round(b.geometry[0][1], 1),  # y_min (top edge)
                            b.geometry[0][0],            # x_min (left edge)
                        ),
                    )
                    block_texts: list[str] = []
                    for block in sorted_blocks:
                        block_lines = [
                            " ".join(w.value for w in line.words)
                            for line in block.lines
                        ]
                        block_text = "\n".join(
                            ln for ln in block_lines if ln.strip()
                        )
                        if block_text.strip():
                            block_texts.append(block_text)

                    # Join blocks with a paragraph break so chunking keeps
                    # semantically distinct paragraphs together.
                    text = "\n\n".join(block_texts)

                    # --- PERBAIKAN TEKS (OLLAMA LLM) ---
                    if text.strip():
                        try:
                            text = self._repair_text_with_ollama(text)
                        except Exception:
                            pass
                    # ------------------------------------

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
        kd_prodi: int | None = None,
    ) -> dict[str, Any]:
        """
        Full pipeline: load file → split → embed → store in ChromaDB.

        Handles OCR for image-based PDFs and manages document versioning
        so that older uploads of the same document are marked as superseded.

        Args:
            file_path: Path to the document file.
            category:  Document category for metadata filtering.
            kd_prodi:  Kode program studi, optional.

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

        # 1.5 Preprocess text
        content = self._preprocessor.preprocess(content, ocr_used=ocr_used)
        if not content.strip():
            raise ValueError(f"File '{file_path.name}' kosong setelah preprocessing.")

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

            meta = {
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
            if kd_prodi is not None:
                meta["kd_prodi"] = kd_prodi
            else:
                meta["kd_prodi"] = 0
                
            metadatas.append(meta)

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
        kd_prodi: int | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for document chunks most similar to the query.

        Results include versioning metadata so the LLM prompt builder can
        signal to the model which documents are current and which are expired.

        Args:
            query:    The user's question.
            top_k:    Number of results to return.
            kd_prodi: Optional program studi filter.
            category: Optional category filter.

        Returns:
            List of dicts with keys: content, source, category,
            relevance_score, document_year, is_latest, ocr_used.
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
            
        retrieve_k = getattr(settings, "TOP_K_RETRIEVE", top_k) if getattr(settings, "USE_RERANKER", False) else top_k

        embeddings_model = self._get_embeddings()
        query_embedding = embeddings_model.embed_query(query)

        where_conditions = []
        if kd_prodi is not None:
            where_conditions.append({
                "$or": [
                    {"kd_prodi": kd_prodi},
                    {"kd_prodi": 0}
                ]
            })
            
        if category is not None:
            where_conditions.append({"category": category})

        where_filter = None
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}

        results = chroma_repo.query(
            query_embedding=query_embedding,
            n_results=retrieve_k,
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

        # ── Reranking Phase ──────────────────────────────────────────
        if getattr(settings, "USE_RERANKER", False) and documents:
            reranker = self._get_reranker()
            if reranker is not None:
                pairs = [[query, doc["content"]] for doc in documents]
                scores = reranker.predict(pairs)
                
                for idx, score in enumerate(scores):
                    # CrossEncoder outputs logits (can be negative or > 1).
                    documents[idx]["relevance_score"] = float(score)
                
                # Sort documents descending by the reranker score
                documents.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Slice to final top_k
        final_docs = documents[:top_k]

        logger.debug(
            "Search returned %d results (from %d retrieved) for query: '%s'", 
            len(final_docs), len(documents), query[:80]
        )
        return final_docs

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _generate_chunk_id(source: str, index: int, content: str) -> str:
        """Generate a deterministic SHA-256-based ID for a chunk."""
        raw = f"{source}::{index}::{content[:100]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_collection_info(self) -> dict[str, Any]:
        """Proxy to the repository's collection info."""
        return chroma_repo.get_collection_info()

    def get_ingested_files(self) -> list[dict[str, Any]]:
        """Proxy to the repository's get_ingested_files."""
        return chroma_repo.get_ingested_files()

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
