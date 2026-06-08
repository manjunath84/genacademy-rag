from genacademy_rag.core.chunker import FixedSizeChunker
from genacademy_rag.core.types import Document


def _doc(text):
    return Document(doc_id="d1", title="README.md", source_type="github", text=text,
                    repo="r", file_path="README.md", commit_hash="abc123")


def test_short_doc_is_one_chunk_with_full_span():
    doc = _doc("line one\nline two\n")
    chunks = FixedSizeChunker(chunk_size=1000, overlap=150).chunk(doc)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "d1::0"
    assert c.text == doc.text
    assert c.citation.char_start == 0
    assert c.citation.char_end == len(doc.text)
    assert c.citation.line_start == 1 and c.citation.line_end == 2
    assert c.citation.commit_hash == "abc123"  # commit_hash chain preserved


def test_long_doc_splits_with_overlap_and_monotonic_spans():
    text = "\n".join(f"sentence number {i} about retrieval" for i in range(400))
    chunks = FixedSizeChunker(chunk_size=300, overlap=60).chunk(_doc(text))
    assert len(chunks) > 1
    # spans cover the document and overlap (next start < prev end)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.citation.char_start < prev.citation.char_end
        assert nxt.ordinal == prev.ordinal + 1


def test_line_numbers_are_one_based_and_correct():
    doc = _doc("a\nb\nc\nd\ne\n" * 50)  # many lines
    chunks = FixedSizeChunker(chunk_size=40, overlap=0).chunk(doc)
    assert chunks[0].citation.line_start == 1
    assert all(c.citation.line_start >= 1 for c in chunks)
