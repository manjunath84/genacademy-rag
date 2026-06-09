"""Pure data types threaded end-to-end. The `commit_hash` chain (fetch → chunk → retrieved
→ eval scorer) lives on Citation so the eval scorer can verify gold provenance."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict


@dataclass(frozen=True)
class Citation:
    doc_id: str
    title: str
    source_type: str  # 'github' | 'pdf' | 'docx' | ...
    # GitHub provenance
    repo: str | None = None
    file_path: str | None = None
    commit_hash: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    # File provenance
    page_or_section: str | None = None
    char_start: int | None = None
    char_end: int | None = None

    def to_metadata(self) -> dict:
        """Flatten for chromadb metadata (str/int/float/bool only; None not allowed)."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_metadata(cls, meta: dict) -> Citation:
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in meta.items() if k in fields})


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # f"{doc_id}::{ordinal}"
    doc_id: str
    ordinal: int
    text: str
    citation: Citation


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    source_type: str
    text: str
    repo: str | None = None
    file_path: str | None = None
    commit_hash: str | None = None
    filename: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    # `score` is the COSINE similarity carried from VectorStore.query (NOT the RRF rank score) — the
    # grader's cosine fallback depends on this. Frozen so it can't be mutated to the wrong signal.
    chunk: Chunk
    score: float


class GraphState(TypedDict, total=False):
    question: str
    retrieved: list  # list[RetrievedChunk]
    answerable: bool
    confidence: int
    used_fallback: bool  # True when the grader degraded to the cosine threshold (observability)
    answer: str
    citations: list  # list[Citation]
    refused: bool
