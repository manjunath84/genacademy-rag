from genacademy_rag.core.chunker import FixedSizeChunker, SectionAwareChunker, build_chunker
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


def test_pdf_pages_become_page_citations():
    text = "page one text\n\f\npage two text\n\f\npage three text"
    doc = Document(doc_id="g", title="g.pdf", source_type="pdf", text=text, filename="g.pdf")
    chunks = FixedSizeChunker(chunk_size=12, overlap=0).chunk(doc)
    pages = {c.citation.page_or_section for c in chunks}
    assert "page 1" in pages and "page 3" in pages


def test_pdf_page_label_follows_chunk_start_across_a_boundary():
    # "AAAA" = page 1, "\f", "BBBB" = page 2. With size=6 the first chunk ("AAAA\fB") straddles the
    # form-feed; the rule is "cite the page where the chunk STARTS", so it must stay page 1, and the
    # next chunk (starting after the \f) must be page 2. Pins the boundary rule against refactors.
    doc = Document(doc_id="g", title="g.pdf", source_type="pdf",
                   text="AAAA\fBBBB", filename="g.pdf")
    chunks = FixedSizeChunker(chunk_size=6, overlap=0).chunk(doc)
    assert chunks[0].text == "AAAA\fB"            # this chunk crosses the page boundary
    assert chunks[0].citation.page_or_section == "page 1"
    assert chunks[1].citation.page_or_section == "page 2"


def test_build_chunker_returns_fixed_by_default():
    # Distinct values per param so a fixed/section param swap fails loudly.
    chunker = build_chunker(
        "fixed",
        chunk_size=900,
        chunk_overlap=100,
        section_max_chars=1700,
        section_overlap=120,
    )

    assert isinstance(chunker, FixedSizeChunker)
    assert chunker.chunk_size == 900
    assert chunker.overlap == 100


def test_build_chunker_returns_section_chunker():
    chunker = build_chunker(
        "section",
        chunk_size=900,
        chunk_overlap=100,
        section_max_chars=1700,
        section_overlap=120,
    )

    assert isinstance(chunker, SectionAwareChunker)
    assert chunker.max_chars == 1700
    assert chunker.overlap == 120


def test_build_chunker_rejects_unknown_name():
    try:
        build_chunker(
            "semantic",
            chunk_size=1000,
            chunk_overlap=150,
            section_max_chars=1500,
            section_overlap=150,
        )
    except ValueError as exc:
        assert "unknown chunker" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_section_chunker_keeps_heading_with_markdown_table():
    text = (
        "# Course\n\n"
        "Intro paragraph.\n\n"
        "## Week 6 Resources\n\n"
        "| Type | Link |\n"
        "| --- | --- |\n"
        "| Video | RAG workshop |\n\n"
        "Closing paragraph.\n"
    )

    chunks = SectionAwareChunker(max_chars=1500, overlap=150).chunk(_doc(text))

    table_chunks = [c for c in chunks if "| Video | RAG workshop |" in c.text]
    assert len(table_chunks) == 1
    assert "## Week 6 Resources" in table_chunks[0].text
    assert table_chunks[0].citation.page_or_section == "section: Course > Week 6 Resources"


def test_section_chunker_starts_new_chunk_at_heading_boundary():
    text = (
        "# Course\n\n"
        "Intro paragraph.\n\n"
        "## Week 6 Resources\n\n"
        "| Type | Link |\n"
        "| --- | --- |\n"
        "| Video | RAG workshop |\n"
    )

    chunks = SectionAwareChunker(max_chars=1500, overlap=150).chunk(_doc(text))

    assert len(chunks) == 2
    assert chunks[0].text == "# Course\n\nIntro paragraph.\n"
    assert chunks[0].citation.page_or_section == "section: Course"
    assert chunks[1].text.startswith("## Week 6 Resources")
    assert "| Video | RAG workshop |" in chunks[1].text
    assert chunks[1].citation.page_or_section == "section: Course > Week 6 Resources"


def test_section_chunker_keeps_fenced_code_block_together_when_under_limit():
    text = (
        "# Lab\n\n"
        "Before code.\n\n"
        "```python\n"
        "print('rag')\n"
        "print('eval')\n"
        "```\n\n"
        "After code.\n"
    )

    chunks = SectionAwareChunker(max_chars=120, overlap=20).chunk(_doc(text))

    code_chunks = [c for c in chunks if "print('rag')" in c.text]
    assert len(code_chunks) == 1
    assert "```python\nprint('rag')\nprint('eval')\n```" in code_chunks[0].text
    assert code_chunks[0].citation.page_or_section == "section: Lab"


def test_section_chunker_splits_oversized_block_with_overlap():
    text = "# Big Section\n\n" + ("alpha beta gamma\n" * 40)

    chunks = SectionAwareChunker(max_chars=120, overlap=30).chunk(_doc(text))

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 120
        assert chunk.citation.page_or_section == "section: Big Section"
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.citation.char_start < prev.citation.char_end
        assert nxt.citation.char_start >= prev.citation.char_start


def test_section_chunker_preserves_monotonic_line_and_char_spans():
    text = (
        "# A\n\n"
        "line one\n"
        "line two\n\n"
        "## B\n\n"
        "line three\n"
        "line four\n"
    )

    chunks = SectionAwareChunker(max_chars=35, overlap=5).chunk(_doc(text))

    assert chunks
    for chunk in chunks:
        assert chunk.text == text[chunk.citation.char_start:chunk.citation.char_end]
        assert chunk.citation.char_start < chunk.citation.char_end
        assert chunk.citation.line_start >= 1
        assert chunk.citation.line_end >= chunk.citation.line_start
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.ordinal == prev.ordinal + 1


def test_section_chunker_short_doc_is_one_full_span_chunk():
    doc = _doc("# One\n\nShort body.\n")

    chunks = SectionAwareChunker(max_chars=1500, overlap=150).chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].text == doc.text
    assert chunks[0].citation.char_start == 0
    assert chunks[0].citation.char_end == len(doc.text)
    assert chunks[0].citation.page_or_section == "section: One"


def test_section_chunker_line_numbers_match_source_exactly():
    # The eval scores gold-span overlap on line numbers, so pin exact values:
    # line 1 "# A", line 3 "intro", line 5 "## B", line 7 "body line".
    text = "# A\n\nintro\n\n## B\n\nbody line\n"

    chunks = SectionAwareChunker(max_chars=1500, overlap=150).chunk(_doc(text))

    assert len(chunks) == 2
    assert chunks[0].citation.line_start == 1
    assert chunks[0].citation.line_end == 3
    assert chunks[1].citation.line_start == 5
    assert chunks[1].citation.line_end == 7


def test_section_chunker_doc_without_headings_chunks_with_no_section_label():
    doc = _doc("para one line.\n\npara two line.\n")

    chunks = SectionAwareChunker(max_chars=1500, overlap=150).chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].text == doc.text
    assert chunks[0].citation.page_or_section is None


def test_section_chunker_whitespace_only_and_empty_docs():
    whitespace = SectionAwareChunker(max_chars=1500, overlap=150).chunk(_doc("   \n\n  \n"))
    assert len(whitespace) == 1
    assert whitespace[0].citation.char_start == 0
    assert whitespace[0].citation.char_end == 8

    assert SectionAwareChunker(max_chars=1500, overlap=150).chunk(_doc("")) == []


def test_section_chunker_rejects_malformed_headings():
    # 7+ marks and missing space after # are not markdown headings.
    text = "####### Seven\n\n#nospace\n\nbody text.\n"

    chunks = SectionAwareChunker(max_chars=1500, overlap=150).chunk(_doc(text))

    assert len(chunks) == 1
    assert chunks[0].citation.page_or_section is None


def test_section_chunker_heading_level_jump_truncates_stack():
    # h1 -> h3 keeps h1 as parent; the later h2 truncates back to h1's child.
    text = "# H1\n\n### H3\n\nbody\n\n## H2\n\nmore\n"

    chunks = SectionAwareChunker(max_chars=1500, overlap=150).chunk(_doc(text))

    labels = [c.citation.page_or_section for c in chunks]
    assert labels == ["section: H1", "section: H1 > H3", "section: H1 > H2"]
