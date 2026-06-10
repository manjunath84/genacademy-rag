# GenAcademy RAG Phase 2 Section-Aware Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disabled-by-default section-aware chunker for markdown-like eval corpus files, then measure its retrieval delta against the fixed-size baseline without mutating baseline artifacts.

**Architecture:** Keep `Chunker` as the pure-core seam and add `SectionAwareChunker` beside `FixedSizeChunker`. Eval ingest gets explicit `--collection`, `--chunker`, `--sqlite-path`, and `--reset-collection` controls so `eval_section` can be regenerated from the same pinned raw docs while the current `eval` collection and SQLite metadata remain intact. Retrieval eval accepts a collection argument and records chunker config in JSON output; rerank stays disabled for the primary chunking delta.

**Tech Stack:** Python 3.12, uv, pytest, ruff, ChromaDB, SQLite, sentence-transformers embeddings, existing deterministic retrieval eval.

---

## Approval Gate

This plan is documentation only. Do not edit implementation code until the user approves this plan or explicitly asks to execute it.

When approved, execute with `superpowers:subagent-driven-development` or `superpowers:executing-plans`:

1. Write the focused failing test.
2. Run the focused test and confirm the expected failure.
3. Implement the smallest source change.
4. Run the focused test and confirm pass.
5. Commit after each task-sized working slice.

Stop after implementation evidence is collected. Do not self-approve or claim the change is review-complete; a separate fresh context reviews the diff.

## Fixed Decisions From Design Review

- First slice is markdown/notebook-text oriented. PDFs and uploads keep default fixed chunking unless the explicit `GENACADEMY_CHUNKER=section` flag is used later.
- Alternate eval ingests use a separate SQLite file by default when `--collection` is not `eval`, with `--sqlite-path` available for explicit control.
- `Citation.page_or_section` stores the full heading path, joined with ` > ` and prefixed with `section: `.
- Primary delta disables rerank to isolate chunking. A combined chunking-plus-rerank run may be added later, but it is not part of this plan.
- Section chunk defaults are `max_chars=1500` and `overlap=150`; structural chunks avoid arbitrary overlap, while oversized-block fallback windows use overlap.

## File Structure

- Modify `src/genacademy_rag/config.py`: add chunker configuration from env.
- Modify `src/genacademy_rag/core/chunker.py`: add `SectionAwareChunker`, shared span helpers, and a pure `build_chunker()` factory.
- Modify `tests/test_config.py`: cover default and env parsing for chunker settings.
- Modify `tests/core/test_chunker.py`: cover section headings, tables, fences, oversized fallback, spans, short docs, and fixed behavior.
- Modify `scripts/ingest_eval_corpus.py`: add CLI flags for collection, chunker, SQLite path, and collection reset.
- Create `tests/eval/test_ingest_eval_corpus_script.py`: offline script wiring tests using fakes.
- Modify `scripts/eval_retrieval.py`: add `--collection`, include collection and chunker settings in JSON config.
- Modify `tests/eval/test_eval_retrieval_script.py`: pin the new eval CLI/config behavior.
- Modify `src/genacademy_rag/web/app.py`: use the chunker factory for upload ingestion, preserving fixed default.
- Modify `tests/web/test_app.py`: adjust fakes only if the chunker factory wiring needs direct coverage.
- Create `eval/phase2-section-aware-chunking-delta.md`: generated after measured baseline and section-aware eval runs.

## Task 0: Branch, Baseline, And Approval Gate

**Files:** none

- [ ] **Step 1: Confirm branch and working tree**

Run:

```bash
git status --short --branch
git rev-parse --short HEAD
```

Expected: branch is the approved implementation branch, not an accidental detached HEAD. There are no uncommitted implementation files before starting.

- [ ] **Step 2: Confirm baseline retrieval eval still reproduces**

Run with rerank disabled:

```bash
GENACADEMY_RERANK_ENABLED=false uv run python scripts/eval_retrieval.py
```

Expected:

```text
RETRIEVAL EVAL  recall@k=0.67  precision@k=0.22  mrr=0.55  (n=12)
```

If this does not reproduce before implementation, stop and report the mismatch.

- [ ] **Step 3: Run the existing test suite**

Run:

```bash
uv run pytest
```

Expected: existing non-integration tests pass. If this fails before implementation, stop and report the pre-existing failure.

No commit for Task 0.

## Task 1: Chunker Settings And Factory

**Files:**

- Modify: `src/genacademy_rag/config.py`
- Modify: `src/genacademy_rag/core/chunker.py`
- Modify: `tests/test_config.py`
- Modify: `tests/core/test_chunker.py`

- [ ] **Step 1: Add failing settings tests**

Add to `tests/test_config.py`:

```python
def test_chunker_defaults_to_fixed(monkeypatch):
    monkeypatch.delenv("GENACADEMY_CHUNKER", raising=False)
    monkeypatch.delenv("GENACADEMY_SECTION_CHUNK_MAX_CHARS", raising=False)
    monkeypatch.delenv("GENACADEMY_SECTION_CHUNK_OVERLAP", raising=False)

    s = Settings.from_env()

    assert s.chunker == "fixed"
    assert s.section_chunk_max_chars == 1500
    assert s.section_chunk_overlap == 150


def test_chunker_env_settings_parse(monkeypatch):
    monkeypatch.setenv("GENACADEMY_CHUNKER", "section")
    monkeypatch.setenv("GENACADEMY_SECTION_CHUNK_MAX_CHARS", "1800")
    monkeypatch.setenv("GENACADEMY_SECTION_CHUNK_OVERLAP", "120")

    s = Settings.from_env()

    assert s.chunker == "section"
    assert s.section_chunk_max_chars == 1800
    assert s.section_chunk_overlap == 120
```

- [ ] **Step 2: Run focused settings tests and verify failure**

Run:

```bash
uv run pytest tests/test_config.py::test_chunker_defaults_to_fixed tests/test_config.py::test_chunker_env_settings_parse -q
```

Expected: FAIL with `AttributeError` for `chunker` or `section_chunk_max_chars`.

- [ ] **Step 3: Add settings fields and env parsing**

Modify `src/genacademy_rag/config.py`.

Add fields to `Settings` after `chunk_overlap`:

```python
    chunker: str
    section_chunk_max_chars: int
    section_chunk_overlap: int
```

Add values to `Settings.from_env()` after `chunk_overlap=...`:

```python
            chunker=os.environ.get("GENACADEMY_CHUNKER", "fixed"),
            section_chunk_max_chars=int(
                os.environ.get("GENACADEMY_SECTION_CHUNK_MAX_CHARS", "1500")
            ),
            section_chunk_overlap=int(
                os.environ.get("GENACADEMY_SECTION_CHUNK_OVERLAP", "150")
            ),
```

- [ ] **Step 4: Add failing factory tests**

Add to `tests/core/test_chunker.py`:

```python
from genacademy_rag.core.chunker import SectionAwareChunker, build_chunker


def test_build_chunker_returns_fixed_by_default():
    chunker = build_chunker(
        "fixed",
        chunk_size=1000,
        chunk_overlap=150,
        section_max_chars=1500,
        section_overlap=150,
    )

    assert isinstance(chunker, FixedSizeChunker)


def test_build_chunker_returns_section_chunker():
    chunker = build_chunker(
        "section",
        chunk_size=1000,
        chunk_overlap=150,
        section_max_chars=1500,
        section_overlap=150,
    )

    assert isinstance(chunker, SectionAwareChunker)


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
```

- [ ] **Step 5: Run focused factory tests and verify failure**

Run:

```bash
uv run pytest tests/core/test_chunker.py::test_build_chunker_returns_fixed_by_default tests/core/test_chunker.py::test_build_chunker_returns_section_chunker tests/core/test_chunker.py::test_build_chunker_rejects_unknown_name -q
```

Expected: FAIL with `ImportError` for `SectionAwareChunker` or `build_chunker`.

- [ ] **Step 6: Add minimal factory and stub section chunker**

Modify `src/genacademy_rag/core/chunker.py`:

```python
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
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run pytest tests/test_config.py tests/core/test_chunker.py::test_build_chunker_returns_fixed_by_default tests/core/test_chunker.py::test_build_chunker_returns_section_chunker tests/core/test_chunker.py::test_build_chunker_rejects_unknown_name -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/genacademy_rag/config.py src/genacademy_rag/core/chunker.py tests/test_config.py tests/core/test_chunker.py
git commit -m "feat: add chunker selection settings"
```

## Task 2: Section-Aware Markdown Chunking

**Files:**

- Modify: `src/genacademy_rag/core/chunker.py`
- Modify: `tests/core/test_chunker.py`

- [ ] **Step 1: Add failing section chunker tests**

Add to `tests/core/test_chunker.py`:

```python
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
```

- [ ] **Step 2: Run focused section tests and verify failure**

Run:

```bash
uv run pytest tests/core/test_chunker.py::test_section_chunker_keeps_heading_with_markdown_table tests/core/test_chunker.py::test_section_chunker_keeps_fenced_code_block_together_when_under_limit tests/core/test_chunker.py::test_section_chunker_splits_oversized_block_with_overlap tests/core/test_chunker.py::test_section_chunker_preserves_monotonic_line_and_char_spans tests/core/test_chunker.py::test_section_chunker_short_doc_is_one_full_span_chunk -q
```

Expected: FAIL because `SectionAwareChunker` still delegates to fixed-size chunking and does not preserve structural sections.

- [ ] **Step 3: Add structural parsing helpers**

Modify `src/genacademy_rag/core/chunker.py`.

Add imports:

```python
from dataclasses import dataclass
```

Add helper data and functions below `Chunker`:

```python
@dataclass(frozen=True)
class _Block:
    start: int
    end: int
    heading_path: tuple[str, ...]


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
```

Then simplify `FixedSizeChunker.chunk()` to use `_line_lookup()` and `_citation_for_span()` while preserving its page label behavior:

```python
        line_at = _line_lookup(text)
```

and:

```python
            citation = _citation_for_span(
                doc,
                start=start,
                end=end,
                line_at=line_at,
                page_or_section=page_or_section,
            )
```

- [ ] **Step 4: Implement markdown block parser**

Add methods to `SectionAwareChunker`:

```python
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
                blocks.append(_Block(start=start, end=end, heading_path=tuple(heading_stack)))
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
```

- [ ] **Step 5: Implement structural chunk emission**

Replace the `SectionAwareChunker.chunk()` stub with:

```python
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
            if block.end - block.start > self.max_chars:
                if pending_start is not None and pending_end is not None:
                    emit(pending_start, pending_end, pending_path)
                    pending_start = None
                    pending_end = None
                    pending_path = ()
                for start, end in self._fallback_windows(block.start, block.end):
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
```

- [ ] **Step 6: Run chunker tests**

Run:

```bash
uv run pytest tests/core/test_chunker.py -q
```

Expected: PASS, including all existing fixed-size tests.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/genacademy_rag/core/chunker.py tests/core/test_chunker.py
git commit -m "feat: add section-aware markdown chunker"
```

## Task 3: Eval Ingest CLI Isolation

**Files:**

- Modify: `scripts/ingest_eval_corpus.py`
- Create: `tests/eval/test_ingest_eval_corpus_script.py`

- [ ] **Step 1: Add failing script wiring tests**

Create `tests/eval/test_ingest_eval_corpus_script.py`:

```python
import sys

import scripts.ingest_eval_corpus as ingest_script
from genacademy_rag.config import Settings
from genacademy_rag.core.types import Document


def _settings(tmp_path):
    return Settings(
        provider="openrouter",
        gen_base_url="https://openrouter.ai/api/v1",
        gen_api_key="",
        gen_model="",
        embed_model="all-MiniLM-L6-v2",
        top_k=5,
        chunk_size=1000,
        chunk_overlap=150,
        chunker="fixed",
        section_chunk_max_chars=1500,
        section_chunk_overlap=150,
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "genacademy.sqlite",
        session_secret="test-secret",
        rerank_enabled=False,
        rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        rerank_local_files_only=True,
        rerank_batch_size=32,
        rerank_pool=0,
        rerank_device=None,
        rerank_cache_dir=None,
    )


def test_ingest_eval_defaults_to_eval_collection_fixed_chunker_and_primary_sqlite(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    state = {}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            state["persist_dir"] = persist_dir
            state["collection"] = collection

        def upsert(self, chunks, embeddings):
            state["upserted"] = [c.chunk_id for c in chunks]

    class _Datastore:
        def __init__(self, path):
            state["sqlite_path"] = path

        def seed_users(self):
            state["seeded"] = True

        def add_document(self, **kwargs):
            state["doc_id"] = kwargs["doc_id"]

        def add_chunks_meta(self, chunks):
            state["chunks_meta"] = [c.chunk_id for c in chunks]

    class _Provider:
        def embed(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(ingest_script.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(ingest_script, "build_provider", lambda s: _Provider())
    monkeypatch.setattr(ingest_script, "ChromaStore", _Store)
    monkeypatch.setattr(ingest_script, "SQLiteDatastore", _Datastore)
    monkeypatch.setattr(
        ingest_script,
        "EVAL_CORPUS",
        [{"owner": "owner", "repo": "repo", "sha": "abc123", "files": [{"path": "README.md", "kind": "markdown"}]}],
    )
    monkeypatch.setattr(ingest_script, "fetch_raw", lambda **kwargs: b"# Title\n\nBody\n")
    monkeypatch.setattr(
        ingest_script,
        "load_markdown",
        lambda **kwargs: Document(
            doc_id="repo/README.md@abc123",
            title="README.md",
            source_type="github",
            text="# Title\n\nBody\n",
            repo="repo",
            file_path="README.md",
            commit_hash="abc123",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["ingest_eval_corpus.py"])

    ingest_script.main()

    assert state["collection"] == "eval"
    assert state["sqlite_path"] == settings.sqlite_path
    assert state["seeded"] is True
    assert state["chunks_meta"] == ["repo/README.md@abc123::0"]


def test_ingest_eval_section_collection_uses_isolated_sqlite_by_default(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    state = {}

    class _Store:
        def __init__(self, *, persist_dir, collection):
            state["collection"] = collection

        def upsert(self, chunks, embeddings):
            state["chunk_texts"] = [c.text for c in chunks]

    class _Datastore:
        def __init__(self, path):
            state["sqlite_path"] = path

        def seed_users(self):
            pass

        def add_document(self, **kwargs):
            state["n_chunks"] = kwargs["n_chunks"]

        def add_chunks_meta(self, chunks):
            state["page_or_section"] = chunks[0].citation.page_or_section

    class _Provider:
        def embed(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(ingest_script.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(ingest_script, "build_provider", lambda s: _Provider())
    monkeypatch.setattr(ingest_script, "ChromaStore", _Store)
    monkeypatch.setattr(ingest_script, "SQLiteDatastore", _Datastore)
    monkeypatch.setattr(ingest_script, "reset_chroma_collection", lambda persist_dir, collection: state.setdefault("reset", collection))
    monkeypatch.setattr(
        ingest_script,
        "EVAL_CORPUS",
        [{"owner": "owner", "repo": "repo", "sha": "abc123", "files": [{"path": "README.md", "kind": "markdown"}]}],
    )
    monkeypatch.setattr(ingest_script, "fetch_raw", lambda **kwargs: b"# Title\n\n| A | B |\n| - | - |\n| C | D |\n")
    monkeypatch.setattr(
        ingest_script,
        "load_markdown",
        lambda **kwargs: Document(
            doc_id="repo/README.md@abc123",
            title="README.md",
            source_type="github",
            text="# Title\n\n| A | B |\n| - | - |\n| C | D |\n",
            repo="repo",
            file_path="README.md",
            commit_hash="abc123",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest_eval_corpus.py", "--collection", "eval_section", "--chunker", "section", "--reset-collection"],
    )

    ingest_script.main()

    assert state["collection"] == "eval_section"
    assert state["reset"] == "eval_section"
    assert state["sqlite_path"] == settings.sqlite_path.with_name("genacademy-eval_section.sqlite")
    assert state["page_or_section"] == "section: Title"
```

- [ ] **Step 2: Run ingest script tests and verify failure**

Run:

```bash
uv run pytest tests/eval/test_ingest_eval_corpus_script.py -q
```

Expected: FAIL because `Settings` requires new chunker fields or `scripts/ingest_eval_corpus.py` does not accept the new CLI arguments.

- [ ] **Step 3: Add CLI parsing, reset helper, chunker selection, and SQLite isolation**

Modify `scripts/ingest_eval_corpus.py`:

```python
import argparse
from pathlib import Path
```

Change imports:

```python
from genacademy_rag.core.chunker import build_chunker
```

Add a reset helper near the top:

```python
def reset_chroma_collection(persist_dir: Path, collection: str) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(collection)
    except ValueError:
        return
```

Add a parser helper:

```python
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="eval")
    parser.add_argument("--chunker", choices=("fixed", "section"), default=None)
    parser.add_argument("--sqlite-path", type=Path)
    parser.add_argument("--reset-collection", action="store_true")
    return parser.parse_args()
```

Replace the start of `main()` with:

```python
def main():
    args = _parse_args()
    s = Settings.from_env()
    chunker_name = args.chunker or s.chunker
    sqlite_path = args.sqlite_path or s.sqlite_path
    if args.collection != "eval" and args.sqlite_path is None:
        sqlite_path = s.sqlite_path.with_name(f"{s.sqlite_path.stem}-{args.collection}.sqlite")
    if args.reset_collection:
        reset_chroma_collection(s.chroma_dir, args.collection)
    provider = build_provider(s)
    store = ChromaStore(persist_dir=s.chroma_dir, collection=args.collection)
    ds = SQLiteDatastore(sqlite_path)
    ds.seed_users()
    pipe = IngestPipeline(
        chunker=build_chunker(
            chunker_name,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            section_max_chars=s.section_chunk_max_chars,
            section_overlap=s.section_chunk_overlap,
        ),
        provider=provider,
        store=store,
        datastore=ds,
    )
```

Replace the final print with:

```python
    print(
        f"ingested {len(docs)} docs -> {n} chunks into "
        f"{s.chroma_dir} collection={args.collection} chunker={chunker_name} sqlite={sqlite_path}"
    )
```

- [ ] **Step 4: Run ingest script tests**

Run:

```bash
uv run pytest tests/eval/test_ingest_eval_corpus_script.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/ingest_eval_corpus.py tests/eval/test_ingest_eval_corpus_script.py
git commit -m "feat: isolate alternate eval chunking ingest"
```

## Task 4: Retrieval Eval Collection And Config Snapshot

**Files:**

- Modify: `scripts/eval_retrieval.py`
- Modify: `tests/eval/test_eval_retrieval_script.py`

- [ ] **Step 1: Update failing eval script test**

Modify the `Settings(...)` construction in `tests/eval/test_eval_retrieval_script.py` to include:

```python
        chunker="section",
        section_chunk_max_chars=1500,
        section_chunk_overlap=150,
```

Change the argv line:

```python
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_retrieval.py", "--collection", "eval_section", "--json-out", str(out)],
    )
```

Change assertions:

```python
    assert state["collection"] == "eval_section"
    assert payload["config"]["collection"] == "eval_section"
    assert payload["config"]["chunker"] == "section"
    assert payload["config"]["section_chunk_max_chars"] == 1500
    assert payload["config"]["section_chunk_overlap"] == 150
```

- [ ] **Step 2: Run eval script test and verify failure**

Run:

```bash
uv run pytest tests/eval/test_eval_retrieval_script.py -q
```

Expected: FAIL because `scripts/eval_retrieval.py` does not accept `--collection` and does not include chunker fields in config.

- [ ] **Step 3: Add collection argument and config fields**

Modify `scripts/eval_retrieval.py`.

Change `_config_snapshot`:

```python
def _config_snapshot(settings: Settings, *, collection: str) -> dict:
    return {
        "collection": collection,
        "top_k": settings.top_k,
        "candidate_k": DEFAULT_CANDIDATE_K,
        "embed_model": settings.embed_model,
        "chunker": settings.chunker,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "section_chunk_max_chars": settings.section_chunk_max_chars,
        "section_chunk_overlap": settings.section_chunk_overlap,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_model": settings.rerank_model,
        "rerank_pool": settings.rerank_pool,
        "rerank_device": settings.rerank_device or "",
        "rerank_batch_size": settings.rerank_batch_size,
        "rerank_local_files_only": settings.rerank_local_files_only,
    }
```

Add parser arg:

```python
    ap.add_argument("--collection", default="eval")
```

Change store construction:

```python
    store = ChromaStore(persist_dir=s.chroma_dir, collection=args.collection)
```

Change JSON payload creation:

```python
        payload = build_retrieval_eval_payload(
            metrics=agg,
            rows=scores,
            config=_config_snapshot(s, collection=args.collection),
        )
```

- [ ] **Step 4: Run eval script test**

Run:

```bash
uv run pytest tests/eval/test_eval_retrieval_script.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/eval_retrieval.py tests/eval/test_eval_retrieval_script.py
git commit -m "feat: evaluate alternate retrieval collections"
```

## Task 5: Default App Upload Wiring

**Files:**

- Modify: `src/genacademy_rag/web/app.py`
- Modify: `tests/web/test_app.py` if needed

- [ ] **Step 1: Change upload ingest chunker wiring**

Modify `src/genacademy_rag/web/app.py`.

Change the chunker import:

```python
    from genacademy_rag.core.chunker import build_chunker
```

Change the ingest pipeline construction:

```python
    pipe = IngestPipeline(
        chunker=build_chunker(
            s.chunker,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            section_max_chars=s.section_chunk_max_chars,
            section_overlap=s.section_chunk_overlap,
        ),
        provider=provider,
        store=serving,
        datastore=datastore,
    )
```

Default behavior remains fixed because `GENACADEMY_CHUNKER` defaults to `fixed`.

- [ ] **Step 2: Run focused web tests**

Run:

```bash
uv run pytest tests/web/test_app.py -q
```

Expected: PASS. If a fake `Settings(...)` construction fails, add the three chunker fields with default values in that test only.

- [ ] **Step 3: Commit**

Run:

```bash
git add src/genacademy_rag/web/app.py tests/web/test_app.py
git commit -m "feat: wire configurable upload chunker"
```

## Task 6: Eval Runs And Delta Report

**Files:**

- Create: `eval/phase2-section-aware-chunking-delta.md`
- Local ignored outputs: `eval/runs/phase2-section-baseline.json`, `eval/runs/phase2-section-aware.json`

- [ ] **Step 1: Run fixed baseline JSON eval with rerank disabled**

Run:

```bash
GENACADEMY_RERANK_ENABLED=false GENACADEMY_CHUNKER=fixed uv run python scripts/eval_retrieval.py \
  --collection eval \
  --json-out eval/runs/phase2-section-baseline.json
```

Expected: printed metrics match the known fixed baseline unless the eval corpus has intentionally changed:

```text
RETRIEVAL EVAL  recall@k=0.67  precision@k=0.22  mrr=0.55  (n=12)
```

- [ ] **Step 2: Ingest section-aware alternate eval collection**

Run:

```bash
GENACADEMY_RERANK_ENABLED=false GENACADEMY_CHUNKER=section uv run python scripts/ingest_eval_corpus.py \
  --collection eval_section \
  --chunker section \
  --reset-collection
```

Expected: output says `collection=eval_section`, `chunker=section`, and the SQLite path ends with `genacademy-eval_section.sqlite`.

- [ ] **Step 3: Run section-aware JSON eval with rerank disabled**

Run:

```bash
GENACADEMY_RERANK_ENABLED=false GENACADEMY_CHUNKER=section uv run python scripts/eval_retrieval.py \
  --collection eval_section \
  --json-out eval/runs/phase2-section-aware.json
```

Expected: retrieval eval completes and writes JSON. Do not edit the gold set if any metric is worse.

- [ ] **Step 4: Generate the delta report from measured JSON**

Read the two JSON files and create `eval/phase2-section-aware-chunking-delta.md` with this structure, replacing the numeric values with measured values:

```markdown
# Phase 2 Section-Aware Chunking Delta

**Date:** 2026-06-10
**Primary comparison:** fixed-size chunking vs section-aware chunking
**Rerank:** disabled for both primary runs
**Gold set:** `src/genacademy_rag/eval/gold/gold_set.yaml`
**Baseline collection:** `eval`
**Candidate collection:** `eval_section`

## Summary

| Run | recall@k | precision@k | MRR | mean retrieval ms | p50 retrieval ms | p95 retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed baseline | measured | measured | measured | measured | measured | measured |
| section-aware | measured | measured | measured | measured | measured | measured |

## Corpus Shape

| Run | collection | chunker | chunk count |
| --- | --- | --- | ---: |
| fixed baseline | eval | fixed | measured |
| section-aware | eval_section | section | measured |

## Per-Question Movement

| ID | Category | Baseline recall | Section recall | Baseline MRR | Section MRR | Movement |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| q1 | answerable | measured | measured | measured | measured | measured |

## Chunking-Stress Questions

| ID | Expected pressure | Movement |
| --- | --- | --- |
| q5 | compact markdown table row with section context | measured |
| q7 | prerequisite table split across fixed windows | measured |
| q8 | Week 6 resource table truncation | measured |
| q9 | multi-document top-k pressure | measured |
| q10 | multi-document top-k pressure | measured |

## Interpretation

State whether section-aware chunking improved, hurt, or left unchanged the known chunking-boundary failures. Keep the small-N caveat explicit because the retrieval eval has 12 retrieval-scored questions.

## Recommendation

Choose one:

- Enable section-aware chunking only for eval/demo ingest.
- Keep it implemented but disabled by default.
- Drop it from the demo path and keep fixed-size chunking.
```

The final committed report must contain only measured values, not the word `measured`.

- [ ] **Step 5: Verify report has no unfinished markers**

Run:

```bash
rg -n "measured|place[h]older|T[B]D|TO[D]O" eval/phase2-section-aware-chunking-delta.md
```

Expected: no matches.

- [ ] **Step 6: Commit**

Run:

```bash
git add eval/phase2-section-aware-chunking-delta.md
git commit -m "docs: report section-aware chunking eval delta"
```

## Task 7: Final Verification

**Files:** all implementation files touched by earlier tasks

- [ ] **Step 1: Run lint**

Run:

```bash
uv run ruff check .
```

Expected: PASS.

- [ ] **Step 2: Run tests**

Run:

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 3: Confirm no baseline eval mutation**

Run:

```bash
GENACADEMY_RERANK_ENABLED=false GENACADEMY_CHUNKER=fixed uv run python scripts/eval_retrieval.py --collection eval
```

Expected:

```text
RETRIEVAL EVAL  recall@k=0.67  precision@k=0.22  mrr=0.55  (n=12)
```

If the fixed baseline changed, stop and investigate before opening a PR.

- [ ] **Step 4: Review changed files**

Run:

```bash
git status --short
git diff --stat HEAD
```

Expected: no uncommitted changes after the final report commit, or only ignored raw eval files under `eval/runs/`.

## Self-Review Checklist

- Spec coverage: this plan covers section-aware chunking, citation spans, eval isolation, collection CLI controls, config defaults, JSON config reporting, metric report, and baseline preservation.
- Scope control: this plan does not add query rewriting, adjacent-chunk stitching, Pinecone, Nebius changes, deploy work, or UI changes beyond upload chunker wiring.
- Unfinished-marker scan: before implementation, run `rg -n "T[B]D|TO[D]O|place[h]older" docs/superpowers/plans/2026-06-10-genacademy-rag-phase2-section-aware-chunking.md` and fix any accidental matches.
- Type consistency: use `section_chunk_max_chars` and `section_chunk_overlap` consistently across `Settings`, tests, CLI config snapshots, and factory arguments.
