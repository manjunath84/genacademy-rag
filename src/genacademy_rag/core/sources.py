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
