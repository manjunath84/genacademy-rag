"""Chunker seam. FixedSizeChunker = character windows with overlap, capturing exact char spans
and 1-based line spans for citations. Char-based (≈250 tok at size 1000) respects the embedder's
256-token cap; token-exact/section-aware chunking is the Phase-2 eval axis."""
from __future__ import annotations

from typing import Protocol

from genacademy_rag.core.types import Chunk, Citation, Document


class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        n = len(text)
        if n == 0:
            return []
        # Precompute char-offset -> line number (1-based) for span citations.
        line_at = [1] * (n + 1)
        line = 1
        for i, ch in enumerate(text):
            line_at[i] = line
            if ch == "\n":
                line += 1
        line_at[n] = line

        has_pages = "\f" in text
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        ordinal = 0
        start = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            piece = text[start:end]
            page_or_section = (
                f"page {text.count(chr(12), 0, start) + 1}" if has_pages else None
            )
            citation = Citation(
                doc_id=doc.doc_id, title=doc.title, source_type=doc.source_type,
                repo=doc.repo, file_path=doc.file_path, commit_hash=doc.commit_hash,
                line_start=line_at[start], line_end=line_at[max(start, end - 1)],
                char_start=start, char_end=end,
                page_or_section=page_or_section,
            )
            chunks.append(Chunk(chunk_id=f"{doc.doc_id}::{ordinal}", doc_id=doc.doc_id,
                                ordinal=ordinal, text=piece, citation=citation))
            ordinal += 1
            if end == n:
                break
            start += step
        return chunks


class SectionAwareChunker:
    def __init__(self, max_chars: int = 1500, overlap: int = 150):
        if overlap >= max_chars:
            raise ValueError("overlap must be < max_chars")
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        return FixedSizeChunker(self.max_chars, self.overlap).chunk(doc)


def build_chunker(
    kind: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    section_max_chars: int,
    section_overlap: int,
) -> Chunker:
    if kind == "fixed":
        return FixedSizeChunker(chunk_size, chunk_overlap)
    if kind == "section":
        return SectionAwareChunker(section_max_chars, section_overlap)
    raise ValueError(f"unknown chunker {kind!r}; expected 'fixed' or 'section'")
