"""PDF → Document (production corpus). pypdf only — spike confirmed the guidebook needs no OCR
(printable ratio 0.994). Per-page text concatenated; page boundaries kept for page citations via
a form-feed marker the chunker preserves in char spans."""
from __future__ import annotations

import hashlib
import io

from pypdf import PdfReader

from genacademy_rag.core.types import Document


def load_pdf_bytes(*, filename: str, raw_bytes: bytes, uploaded_by: str | None = None) -> Document:
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\f\n".join(pages)  # form-feed separates pages
    doc_id = "pdf/" + hashlib.sha256(raw_bytes).hexdigest()[:12]
    return Document(doc_id=doc_id, title=filename, source_type="pdf", text=text,
                    filename=filename)
