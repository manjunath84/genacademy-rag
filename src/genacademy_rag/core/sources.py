"""Pure presentation helpers for the answer card.

Bucket grader confidence, build verification URLs, and merge retrieved citations
into deduped source rows. No web imports: this stays unit-testable offline.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from genacademy_rag.core.types import Citation, RetrievedChunk

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


@dataclass(frozen=True)
class SourceView:
    """One deduped source row on the answer card."""

    title: str
    url: str | None
    range_label: str
    meta_label: str
    snippet: str


def merge_citations(retrieved: list[RetrievedChunk]) -> list[SourceView]:
    """Collapse per-chunk citations into deduped source rows."""
    groups: dict[tuple, list[tuple[int, RetrievedChunk]]] = {}
    for rank, rc in enumerate(retrieved):
        cit = rc.chunk.citation
        if cit.repo:
            key = ("gh", cit.repo, cit.file_path, cit.commit_hash)
        else:
            key = ("doc", cit.doc_id)
        groups.setdefault(key, []).append((rank, rc))

    ranked_views: list[tuple[int, SourceView]] = []
    for key, members in groups.items():
        if key[0] == "gh":
            ranked_views.extend(_github_views(members))
        else:
            ranked_views.append(_file_view(members))
    ranked_views.sort(key=lambda rv: rv[0])
    return [view for _, view in ranked_views]


def _github_views(members: list[tuple[int, RetrievedChunk]]) -> list[tuple[int, SourceView]]:
    lined = [m for m in members if m[1].chunk.citation.line_start is not None]
    unlined = [m for m in members if m[1].chunk.citation.line_start is None]

    spans: list[dict] = []
    for rank, rc in sorted(lined, key=lambda m: m[1].chunk.citation.line_start or 0):
        cit = rc.chunk.citation
        start = cit.line_start
        end = cit.line_end if cit.line_end is not None else start
        if start is None:
            continue
        if spans and start <= spans[-1]["end"] + 1:
            cur = spans[-1]
            cur["end"] = max(cur["end"], end)
            if rank < cur["rank"]:
                cur["rank"], cur["rc"] = rank, rc
        else:
            spans.append({"start": start, "end": end, "rank": rank, "rc": rc})

    out: list[tuple[int, SourceView]] = []
    for span in spans:
        cit = span["rc"].chunk.citation
        merged_cit = replace(cit, line_start=span["start"], line_end=span["end"])
        out.append(
            (
                span["rank"],
                SourceView(
                    title=cit.title,
                    url=github_url(merged_cit),
                    range_label=f"lines {span['start']}–{span['end']}",
                    meta_label=f"{cit.repo} @ {(cit.commit_hash or '')[:7]}",
                    snippet=span["rc"].chunk.text[:SNIPPET_CHARS],
                ),
            )
        )
    for rank, rc in unlined:
        cit = rc.chunk.citation
        out.append(
            (
                rank,
                SourceView(
                    title=cit.title,
                    url=github_url(cit),
                    range_label="",
                    meta_label=f"{cit.repo} @ {(cit.commit_hash or '')[:7]}",
                    snippet=rc.chunk.text[:SNIPPET_CHARS],
                ),
            )
        )
    return out


def _file_view(members: list[tuple[int, RetrievedChunk]]) -> tuple[int, SourceView]:
    rank, rc = min(members, key=lambda m: m[0])
    cit = rc.chunk.citation
    return (
        rank,
        SourceView(
            title=cit.title,
            url=f"/documents/{cit.doc_id}/file",
            range_label=cit.page_or_section or "",
            meta_label="uploaded document",
            snippet=rc.chunk.text[:SNIPPET_CHARS],
        ),
    )
