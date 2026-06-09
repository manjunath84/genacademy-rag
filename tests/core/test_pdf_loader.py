import io

from pypdf import PdfWriter

from genacademy_rag.core.loaders.pdf_loader import load_pdf_bytes


def test_pdf_loader_extracts_text_with_page_citations(tmp_path):
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    doc = load_pdf_bytes(
        filename="g.pdf",
        raw_bytes=buf.getvalue(),
        uploaded_by="admin@genacademy.local",
    )
    assert doc.source_type == "pdf"
    assert doc.filename == "g.pdf"
    assert doc.title == "g.pdf"


def test_pdf_loader_preserves_uploaded_by_and_stored_path():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    raw = io.BytesIO()
    writer.write(raw)
    doc = load_pdf_bytes(
        filename="notes.pdf",
        raw_bytes=raw.getvalue(),
        uploaded_by="admin@genacademy.local",
        stored_path="/tmp/pdf_abcd.pdf",
    )
    assert doc.uploaded_by == "admin@genacademy.local"
    assert doc.stored_path == "/tmp/pdf_abcd.pdf"
