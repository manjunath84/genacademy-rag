"""PDF → Document (production corpus). pypdf only — spike confirmed the guidebook needs no OCR
(printable ratio 0.994). Per-page text concatenated; page boundaries kept for page citations via
a form-feed marker the chunker preserves in char spans."""
from __future__ import annotations

import hashlib
import io
import logging

from pypdf import PdfReader

from genacademy_rag.core.types import Document

logger = logging.getLogger(__name__)


def load_pdf_bytes(
    *,
    filename: str,
    raw_bytes: bytes,
    uploaded_by: str | None = None,
    stored_path: str | None = None,
) -> Document:
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [(p.extract_text() or "") for p in reader.pages]
    n_empty = sum(1 for p in pages if not p.strip())
    if n_empty:
        # A page pypdf can't extract (scanned image, encoding fault) becomes "" and ingests as a
        # hollow chunk. Surface it so an extraction failure isn't mistaken for a retrieval miss.
        logger.warning("%s: %d/%d pages extracted no text", filename, n_empty, len(pages))
    text = "\n\f\n".join(pages)  # form-feed separates pages
    doc_id = "pdf/" + hashlib.sha256(raw_bytes).hexdigest()[:12]
    return Document(
        doc_id=doc_id,
        title=filename,
        source_type="pdf",
        text=text,
        filename=filename,
        uploaded_by=uploaded_by,
        stored_path=stored_path,
    )
