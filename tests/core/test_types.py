from genacademy_rag.core.types import Chunk, Citation, Document


def test_github_citation_round_trips_through_chunk():
    cit = Citation(
        doc_id="d1", title="README.md", source_type="github",
        repo="awesome-agentic-ai-resources", file_path="README.md",
        commit_hash="5dfb8691180dc4956107e86839998ba3a2ebd94f",
        line_start=10, line_end=18,
    )
    chunk = Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0, text="hello", citation=cit)
    assert chunk.citation.commit_hash.startswith("5dfb869")
    assert chunk.citation.line_end == 18


def test_citation_to_flat_metadata_omits_none():
    cit = Citation(doc_id="d1", title="g.pdf", source_type="pdf",
                   page_or_section="p3", char_start=0, char_end=99)
    flat = cit.to_metadata()
    assert flat["source_type"] == "pdf"
    assert flat["page_or_section"] == "p3"
    assert "repo" not in flat  # None values dropped (chromadb metadata cannot be None)


def test_document_carries_commit_hash():
    doc = Document(doc_id="d1", title="README.md", source_type="github", text="x",
                   repo="r", file_path="README.md", commit_hash="abc123")
    assert doc.commit_hash == "abc123"


def test_citation_round_trips_via_metadata_github():
    original = Citation(
        doc_id="d1", title="README.md", source_type="github",
        repo="awesome-agentic-ai-resources", file_path="README.md",
        commit_hash="5dfb8691180dc4956107e86839998ba3a2ebd94f",
        line_start=10, line_end=18,
    )
    assert Citation.from_metadata(original.to_metadata()) == original


def test_citation_round_trips_via_metadata_pdf_with_zero_offset():
    # char_start=0 must survive the `if v is not None` filter (zero is not None).
    original = Citation(
        doc_id="d2", title="guide.pdf", source_type="pdf",
        page_or_section="p3", char_start=0, char_end=99,
    )
    assert Citation.from_metadata(original.to_metadata()) == original
