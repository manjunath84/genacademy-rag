"""Pure presentation helpers for the answer card.

Bucket grader confidence, build verification URLs, and merge retrieved citations
into deduped source rows. No web imports: this stays unit-testable offline.
"""
from __future__ import annotations

from genacademy_rag.core.types import Citation

GITHUB_OWNER = "The-Gen-Academy"
SNIPPET_CHARS = 240


def confidence_bucket(confidence: int) -> str:
    """Map the grader's 1-5 answerability signal to an honest display label."""
    if confidence <= 2:
        return "low"
    if confidence == 3:
        return "medium"
    return "high"


def github_url(citation: Citation) -> str | None:
    """Pinned-commit GitHub URL at the cited line range; None for non-GitHub sources."""
    if not (citation.repo and citation.commit_hash and citation.file_path):
        return None
    url = (
        f"https://github.com/{GITHUB_OWNER}/{citation.repo}/blob/"
        f"{citation.commit_hash}/{citation.file_path}"
    )
    if citation.line_start is not None and citation.line_end is not None:
        url += f"#L{citation.line_start}-L{citation.line_end}"
    return url
