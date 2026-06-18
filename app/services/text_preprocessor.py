import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Multi-stage text preprocessing for academic documents."""

    def preprocess(self, text: str, *, ocr_used: bool = False) -> str:
        """
        Multi-stage text preprocessing pipeline.
        
        Applied after document extraction/OCR and before text chunking.
        Each stage is independent.
        
        Args:
            text: Raw extracted text from PDF/TXT/MD.
            ocr_used: Whether OCR was applied (enables OCR-specific fixes).
        
        Returns:
            Cleaned, normalized text ready for chunking.
        """
        logger.info("Starting preprocessing pipeline (ocr_used=%s)…", ocr_used)
        original_len = len(text)

        # Stage 1: Unicode normalization
        text = self._normalize_unicode(text)

        # Stage 2: Table normalization (must run before whitespace cleanup)
        text = self._normalize_tables(text)

        # Stage 3: Whitespace cleanup
        text = self._clean_whitespace(text)

        # Stage 4: OCR artifact correction (conditional)
        if ocr_used:
            text = self._fix_ocr_artifacts(text)

        # Stage 5: Header/footer removal
        text = self._remove_headers_footers(text)

        # Stage 6: Section-aware restructuring
        text = self._restructure_sections(text)

        # Stage 7: Final cleanup
        text = self._final_cleanup(text)

        logger.info(
            "Preprocessing complete: %d → %d chars (%.1f%% reduction)",
            original_len,
            len(text),
            (1 - len(text) / original_len) * 100 if original_len else 0,
        )
        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to NFC and remove invisible characters."""
        text = unicodedata.normalize("NFC", text)
        # Remove BOM, zero-width spaces, and other invisible chars
        text = re.sub(r"[\ufeff\u200b\u200c\u200d\u2060\ufffe]", "", text)
        # Normalize fancy quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace("–", "-").replace("—", "-")
        # Remove control chars except newline and tab
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text

    def _clean_whitespace(self, text: str) -> str:
        """Normalize whitespace: collapse runs, fix line breaks, rejoin hyphens."""
        # Rejoin soft-hyphenated words across line breaks
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        # Collapse multiple spaces (but not newlines) into one
        text = re.sub(r"[^\S\n]+", " ", text)
        # Remove trailing spaces per line
        text = re.sub(r" +\n", "\n", text)
        # Collapse 3+ consecutive newlines into 2 (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _fix_ocr_artifacts(self, text: str) -> str:
        """Fix common OCR misrecognition patterns from doctr."""
        # Remove lines that are just repeated symbols (OCR noise)
        text = re.sub(r"^[#*?_=\-\.]{5,}$", "", text, flags=re.MULTILINE)
        # Fix zero-in-word → 'O' (e.g., Pr0gram → Program)
        text = re.sub(r"(?<=[a-zA-Z])0(?=[a-zA-Z])", "O", text)
        # Remove isolated single characters that are likely OCR noise
        text = re.sub(r"(?<=\s)[^\w\s](?=\s)", "", text)
        # Collapse multiple punctuation
        text = re.sub(r"([.!?]){3,}", r"\1", text)
        return text

    def _remove_headers_footers(self, text: str) -> str:
        """Remove repeating header/footer lines across pages."""
        pages = text.split("\n\n")
        if len(pages) < 3:
            return text  # Too few pages to detect patterns

        # Count line frequency across pages
        line_freq: dict[str, int] = {}
        for page in pages:
            seen = set()
            for line in page.strip().split("\n"):
                normalized = line.strip().lower()
                # Skip very short lines and pure numbers (page numbers)
                if len(normalized) < 3:
                    continue
                # Normalize page numbers: "halaman 5" → "halaman X"
                normalized = re.sub(r"\d+", "X", normalized)
                if normalized not in seen:
                    line_freq[normalized] = line_freq.get(normalized, 0) + 1
                    seen.add(normalized)

        threshold = len(pages) * 0.5
        repeat_patterns = {pat for pat, count in line_freq.items() if count >= threshold}

        if not repeat_patterns:
            return text

        # Remove matching lines
        cleaned_lines = []
        for line in text.split("\n"):
            normalized = re.sub(r"\d+", "X", line.strip().lower())
            if normalized not in repeat_patterns:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _normalize_tables(self, text: str) -> str:
        """Detect and normalize broken table formatting from PDF extraction."""
        lines = text.split("\n")
        result = []
        table_buffer = []
        in_table = False

        for line in lines:
            # Heuristic: line with 3+ segments separated by 2+ spaces = table row
            segments = re.split(r"\s{2,}", line.strip())
            is_table_line = len(segments) >= 3 and all(len(s) < 50 for s in segments)

            if is_table_line:
                if not in_table:
                    in_table = True
                    result.append("")  # blank line before table
                table_buffer.append(" | ".join(segments))
            else:
                if in_table:
                    result.extend(table_buffer)
                    result.append("")  # blank line after table
                    table_buffer = []
                    in_table = False
                result.append(line)

        if table_buffer:
            result.extend(table_buffer)

        return "\n".join(result)

    def _restructure_sections(self, text: str) -> str:
        """Add clear separators between detected document sections."""
        # Pattern untuk heading dokumen akademik Indonesia
        section_patterns = [
            r"^(BAB\s+[IVXLCDM]+\.?\s*.+)$",           # BAB I, BAB II, dll
            r"^(\d+\.\d+\.?\s+[A-Z].+)$",                # 1.1 Pendahuluan
            r"^([A-Z][A-Z\s]{3,})$",                      # KATA PENGANTAR, dll
            r"^(Pasal\s+\d+)$",                           # Pasal 1, Pasal 2
        ]
        combined = "|".join(f"({p})" for p in section_patterns)

        lines = text.split("\n")
        result = []
        for line in lines:
            if re.match(combined, line.strip(), re.IGNORECASE):
                result.append("\n\n")  # Extra separator before heading
                result.append(line)
            else:
                result.append(line)

        return "\n".join(result)

    def _final_cleanup(self, text: str) -> str:
        """Final pass: remove empty sections, normalize paragraph spacing."""
        # Remove lines with only whitespace
        text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)
        # Collapse 3+ newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Ensure text doesn't start/end with excessive whitespace
        return text.strip()
