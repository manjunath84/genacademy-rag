"""Chunker seam. FixedSizeChunker = character windows with overlap; SectionAwareChunker =
markdown-heading-bounded blocks (tables/fences kept intact, fixed-window fallback with overlap
for oversized blocks). Both capture exact char spans and 1-based line spans for citations.
Fixed chunks (~250 tok at size 1000) fit the embedder's 256-token cap; section chunks may reach
max_chars=1500 (~375 tok) and get tail-truncated by the embedder — a known trade-off measured in
eval/phase2-section-aware-chunking-delta.md."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from genacademy_rag.core.types import Chunk, Citation, Document


class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


@dataclass(frozen=True)
class _Block:
    start: int
    end: int
    heading_path: tuple[str, ...]
    is_heading: bool = False


def _line_lookup(text: str) -> list[int]:
    line_at = [1] * (len(text) + 1)
    line = 1
    for i, ch in enumerate(text):
        line_at[i] = line
        if ch == "\n":
            line += 1
    line_at[len(text)] = line
    return line_at


def _citation_for_span(
    doc: Document,
    *,
    start: int,
    end: int,
    line_at: list[int],
    page_or_section: str | None,
) -> Citation:
    return Citation(
        doc_id=doc.doc_id,
        title=doc.title,
        source_type=doc.source_type,
        repo=doc.repo,
        file_path=doc.file_path,
        commit_hash=doc.commit_hash,
        line_start=line_at[start],
        line_end=line_at[max(start, end - 1)],
        char_start=start,
        char_end=end,
        page_or_section=page_or_section,
    )


def _section_label(path: tuple[str, ...]) -> str | None:
    if not path:
        return None
    return "section: " + " > ".join(path)


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
        line_at = _line_lookup(text)

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
            citation = _citation_for_span(
                doc,
                start=start,
                end=end,
                line_at=line_at,
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
        text = doc.text
        if not text:
            return []
        line_at = _line_lookup(text)
        blocks = self._parse_blocks(text)
        chunks: list[Chunk] = []
        pending_start: int | None = None
        pending_end: int | None = None
        pending_path: tuple[str, ...] = ()

        def emit(start: int, end: int, path: tuple[str, ...]) -> None:
            ordinal = len(chunks)
            citation = _citation_for_span(
                doc,
                start=start,
                end=end,
                line_at=line_at,
                page_or_section=_section_label(path),
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{ordinal}",
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    text=text[start:end],
                    citation=citation,
                )
            )

        for block in blocks:
            if block.is_heading and pending_start is not None and pending_end is not None:
                emit(pending_start, pending_end, pending_path)
                pending_start = None
                pending_end = None
                pending_path = ()

            if block.end - block.start > self.max_chars:
                window_start = block.start
                if pending_start is not None and pending_end is not None:
                    window_start = pending_start
                    pending_start = None
                    pending_end = None
                    pending_path = ()
                for start, end in self._fallback_windows(window_start, block.end):
                    emit(start, end, block.heading_path)
                continue

            if pending_start is None:
                pending_start = block.start
                pending_end = block.end
                pending_path = block.heading_path
                continue

            candidate_end = block.end
            if candidate_end - pending_start <= self.max_chars:
                pending_end = candidate_end
                if block.heading_path:
                    pending_path = block.heading_path
                continue

            emit(pending_start, pending_end, pending_path)
            pending_start = block.start
            pending_end = block.end
            pending_path = block.heading_path

        if pending_start is not None and pending_end is not None:
            emit(pending_start, pending_end, pending_path)

        return chunks

    def _parse_blocks(self, text: str) -> list[_Block]:
        lines = text.splitlines(keepends=True)
        offsets: list[int] = []
        pos = 0
        for line in lines:
            offsets.append(pos)
            pos += len(line)

        blocks: list[_Block] = []
        heading_stack: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue

            heading = self._heading(line)
            if heading is not None:
                level, title = heading
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(title)
                start = offsets[i]
                end = start + len(line)
                blocks.append(
                    _Block(
                        start=start,
                        end=end,
                        heading_path=tuple(heading_stack),
                        is_heading=True,
                    )
                )
                i += 1
                continue

            start_i = i
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence = stripped[:3]
                i += 1
                while i < len(lines):
                    if lines[i].strip().startswith(fence):
                        i += 1
                        break
                    i += 1
            elif "|" in line:
                i += 1
                while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                    i += 1
            else:
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if not next_line.strip():
                        break
                    if self._heading(next_line) is not None:
                        break
                    if next_line.strip().startswith(("```", "~~~")):
                        break
                    if "|" in next_line:
                        break
                    i += 1

            start = offsets[start_i]
            end = offsets[i - 1] + len(lines[i - 1])
            blocks.append(_Block(start=start, end=end, heading_path=tuple(heading_stack)))

        if not blocks and text:
            blocks.append(_Block(start=0, end=len(text), heading_path=()))
        return blocks

    @staticmethod
    def _heading(line: str) -> tuple[int, str] | None:
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            return None
        marks = len(stripped) - len(stripped.lstrip("#"))
        if marks < 1 or marks > 6:
            return None
        if len(stripped) == marks or stripped[marks] != " ":
            return None
        title = stripped[marks:].strip()
        if not title:
            return None
        return marks, title.strip("#").strip()

    def _fallback_windows(self, start: int, end: int) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        step = self.max_chars - self.overlap
        cursor = start
        while cursor < end:
            window_end = min(cursor + self.max_chars, end)
            windows.append((cursor, window_end))
            if window_end == end:
                break
            cursor += step
        return windows


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
