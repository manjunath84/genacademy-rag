"""Tests for the pure answer-card presentation helpers (core/sources.py)."""

from genacademy_rag.core.sources import confidence_bucket, github_url
from genacademy_rag.core.types import Citation


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
