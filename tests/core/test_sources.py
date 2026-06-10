"""Tests for the pure answer-card presentation helpers (core/sources.py)."""

from genacademy_rag.core.sources import SourceView, confidence_bucket, github_url, merge_citations
from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk


def test_confidence_bucket_boundaries():
    assert confidence_bucket(1) == "low"
    assert confidence_bucket(2) == "low"
    assert confidence_bucket(3) == "medium"
    assert confidence_bucket(4) == "high"
    assert confidence_bucket(5) == "high"


def _gh_citation(**overrides):
    base = dict(
        doc_id="d1",
        title="README.md",
        source_type="github",
        repo="awesome-agentic-ai-resources",
        file_path="README.md",
        commit_hash="5dfb8691180dc4956107e86839998ba3a2ebd94f",
        line_start=41,
        line_end=70,
    )
    base.update(overrides)
    return Citation(**base)


def test_github_url_pinned_commit_with_line_anchor():
    url = github_url(_gh_citation())
    assert url == (
        "https://github.com/The-Gen-Academy/awesome-agentic-ai-resources/blob/"
        "5dfb8691180dc4956107e86839998ba3a2ebd94f/README.md#L41-L70"
    )


def test_github_url_without_lines_omits_anchor():
    url = github_url(_gh_citation(line_start=None, line_end=None))
    assert url.endswith("/README.md")
    assert "#L" not in url


def test_github_url_returns_none_for_uploaded_file():
    cit = Citation(
        doc_id="up1",
        title="Week2 Deck",
        source_type="pdf",
        page_or_section="page 3",
    )
    assert github_url(cit) is None


def _rc(text, *, ordinal=0, score=0.8, **cit_overrides):
    cit = _gh_citation(**cit_overrides)
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"{cit.doc_id}::{ordinal}",
            doc_id=cit.doc_id,
            ordinal=ordinal,
            text=text,
            citation=cit,
        ),
        score=score,
    )


def test_merge_overlapping_and_adjacent_ranges_into_one_row():
    # 41-57, 57-65, 64-68, 67-70 -> one row 41-70.
    retrieved = [
        _rc("top ranked chunk", ordinal=0, line_start=57, line_end=65),
        _rc("second", ordinal=1, line_start=41, line_end=57),
        _rc("third", ordinal=2, line_start=64, line_end=68),
        _rc("fourth", ordinal=3, line_start=67, line_end=70),
    ]
    views = merge_citations(retrieved)
    assert len(views) == 1
    v = views[0]
    assert isinstance(v, SourceView)
    assert v.range_label == "lines 41–70"
    assert v.url.endswith("#L41-L70")
    assert v.snippet == "top ranked chunk"
    assert v.meta_label == "awesome-agentic-ai-resources @ 5dfb869"


def test_non_contiguous_ranges_stay_separate_rows():
    retrieved = [
        _rc("intro", ordinal=0, line_start=1, line_end=14),
        _rc("rag section", ordinal=1, line_start=41, line_end=70),
    ]
    views = merge_citations(retrieved)
    assert [v.range_label for v in views] == ["lines 1–14", "lines 41–70"]


def test_rows_ordered_by_best_retrieval_rank():
    retrieved = [
        _rc("rag section", ordinal=0, line_start=41, line_end=70),
        _rc("intro", ordinal=1, line_start=1, line_end=14),
    ]
    views = merge_citations(retrieved)
    assert [v.range_label for v in views] == ["lines 41–70", "lines 1–14"]


def test_uploaded_file_groups_to_one_linked_row():
    cit_kwargs = dict(
        repo=None,
        file_path=None,
        commit_hash=None,
        line_start=None,
        line_end=None,
        doc_id="up1",
        title="Week2 Deck",
        source_type="pdf",
        page_or_section="page 3",
    )
    retrieved = [
        _rc("slide text A", ordinal=0, **cit_kwargs),
        _rc("slide text B", ordinal=1, **cit_kwargs),
    ]
    views = merge_citations(retrieved)
    assert len(views) == 1
    v = views[0]
    assert v.url == "/documents/up1/file"
    assert v.range_label == "page 3"
    assert v.meta_label == "uploaded document"
    assert v.snippet == "slide text A"


def test_snippet_truncated_to_240_chars():
    views = merge_citations([_rc("x" * 1000)])
    assert len(views[0].snippet) == 240
