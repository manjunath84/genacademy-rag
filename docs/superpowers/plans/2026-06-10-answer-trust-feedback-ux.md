# Answer Trust & Feedback UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the answer card so the pipeline's citation discipline is visible — clickable pinned-commit/document source links with merged line ranges and snippets, overview-format answers, an honest Low/Med/High confidence badge, persisted thumbs up/down with admin counts, copy/retry buttons, and an AI-mistake disclaimer.

**Architecture:** Pure core / thin view, unchanged. New pure module `core/sources.py` turns `retrieved` chunks into presentation-ready `SourceView` rows; `QueryResult` carries them; the web layer renders them and adds two routes (`POST /feedback`, `GET /documents/{doc_id}/file`). The eval-facing `QueryResult.citations` is byte-identical to today. Spec: `docs/superpowers/specs/2026-06-10-answer-trust-feedback-ux-design.md`.

**Tech Stack:** Python 3.12 · `uv` · `ruff` · `pytest` · FastAPI + Jinja2 + HTMX (already loaded) + Alpine.js (CDN, locked stack) · SQLite.

**Run all commands from the repo root** (`genacademy-rag/`). Test commands use `uv run pytest …`; lint is `uv run ruff check .`.

---

## Task 0: Branch

- [x] **Step 1: Create the feature branch**

```bash
git checkout -b feat/answer-trust-feedback-ux
```

---

## Task 1: `confidence_bucket` (new `core/sources.py`)

**Files:**
- Create: `src/genacademy_rag/core/sources.py`
- Create: `tests/core/test_sources.py`

- [x] **Step 1: Write the failing tests**

Create `tests/core/test_sources.py`:

```python
"""Tests for the pure answer-card presentation helpers (core/sources.py)."""
from genacademy_rag.core.sources import confidence_bucket


def test_confidence_bucket_boundaries():
    assert confidence_bucket(1) == "low"
    assert confidence_bucket(2) == "low"
    assert confidence_bucket(3) == "medium"
    assert confidence_bucket(4) == "high"
    assert confidence_bucket(5) == "high"
```

- [x] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/core/test_sources.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'genacademy_rag.core.sources'`

- [x] **Step 3: Write the minimal implementation**

Create `src/genacademy_rag/core/sources.py`:

```python
"""Pure presentation helpers for the answer card: bucket grader confidence, build
verification URLs, and merge retrieved citations into deduped source rows. No web
imports — this stays unit-testable offline (AGENTS.md: pure core / thin view)."""
from __future__ import annotations

from genacademy_rag.core.types import Citation

GITHUB_OWNER = "The-Gen-Academy"  # must match the ingest allowlist in core/loaders/__init__.py
SNIPPET_CHARS = 240


def confidence_bucket(confidence: int) -> str:
    """Map the grader's 1-5 answerability bucket to an honest display label.
    Never rendered as a percentage — it is not a calibrated probability."""
    if confidence <= 2:
        return "low"
    if confidence == 3:
        return "medium"
    return "high"
```

- [x] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/core/test_sources.py -v`
Expected: 1 PASS

- [x] **Step 5: Commit**

```bash
git add src/genacademy_rag/core/sources.py tests/core/test_sources.py
git commit -m "feat: add confidence_bucket presentation helper"
```

---

## Task 2: `github_url`

**Files:**
- Modify: `src/genacademy_rag/core/sources.py`
- Modify: `tests/core/test_sources.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/core/test_sources.py`:

```python
from genacademy_rag.core.sources import github_url
from genacademy_rag.core.types import Citation


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
    cit = Citation(doc_id="up1", title="Week2 Deck", source_type="pdf",
                   page_or_section="page 3")
    assert github_url(cit) is None
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/core/test_sources.py -v`
Expected: FAIL — `ImportError: cannot import name 'github_url'`

- [x] **Step 3: Implement**

Append to `src/genacademy_rag/core/sources.py`:

```python
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
```

- [x] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/core/test_sources.py -v`
Expected: 4 PASS

- [x] **Step 5: Commit**

```bash
git add src/genacademy_rag/core/sources.py tests/core/test_sources.py
git commit -m "feat: add pinned-commit github_url builder"
```

---

## Task 3: `SourceView` + `merge_citations`

**Files:**
- Modify: `src/genacademy_rag/core/sources.py`
- Modify: `tests/core/test_sources.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/core/test_sources.py`:

```python
from genacademy_rag.core.sources import SourceView, merge_citations
from genacademy_rag.core.types import Chunk, RetrievedChunk


def _rc(text, *, ordinal=0, score=0.8, **cit_overrides):
    cit = _gh_citation(**cit_overrides)
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"{cit.doc_id}::{ordinal}", doc_id=cit.doc_id,
                    ordinal=ordinal, text=text, citation=cit),
        score=score,
    )


def test_merge_overlapping_and_adjacent_ranges_into_one_row():
    # 41-57, 57-65, 64-68, 67-70 -> one row 41-70 (the user-reported duplicate case)
    retrieved = [
        _rc("top ranked chunk", ordinal=0, line_start=57, line_end=65),
        _rc("second", ordinal=1, line_start=41, line_end=57),
        _rc("third", ordinal=2, line_start=64, line_end=68),
        _rc("fourth", ordinal=3, line_start=67, line_end=70),
    ]
    views = merge_citations(retrieved)
    assert len(views) == 1
    v = views[0]
    assert v.range_label == "lines 41–70"
    assert v.url.endswith("#L41-L70")
    assert v.snippet == "top ranked chunk"  # best-ranked contributor wins the snippet
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
    # rank order, not line order: the rank-0 chunk's row comes first
    assert [v.range_label for v in views] == ["lines 41–70", "lines 1–14"]


def test_uploaded_file_groups_to_one_linked_row():
    cit_kwargs = dict(repo=None, file_path=None, commit_hash=None,
                      line_start=None, line_end=None, doc_id="up1",
                      title="Week2 Deck", source_type="pdf", page_or_section="page 3")
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
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/core/test_sources.py -v`
Expected: FAIL — `ImportError: cannot import name 'SourceView'`

- [x] **Step 3: Implement**

Append to `src/genacademy_rag/core/sources.py` (add `dataclass`/`replace` imports at top):

```python
from dataclasses import dataclass, replace

from genacademy_rag.core.types import Chunk, RetrievedChunk  # noqa: F401 (Chunk for typing context)


@dataclass(frozen=True)
class SourceView:
    """One deduped source row on the answer card."""
    title: str
    url: str | None   # GitHub pinned-commit URL | /documents/{doc_id}/file | None
    range_label: str  # "lines 41–70" | "page 3" | ""
    meta_label: str   # "awesome-agentic-ai-resources @ 5dfb869" | "uploaded document"
    snippet: str      # first SNIPPET_CHARS of the best-ranked contributing chunk


def merge_citations(retrieved: list[RetrievedChunk]) -> list[SourceView]:
    """Collapse per-chunk citations into deduped source rows.

    GitHub chunks group by (repo, file_path, commit_hash) and merge overlapping or
    adjacent line ranges (41-57 + 57-65 + 64-68 + 67-70 -> 41-70); non-contiguous
    ranges in the same file stay separate rows. Uploaded-file chunks group by doc_id
    into one row linking to /documents/{doc_id}/file. Rows are ordered by the best
    retrieval rank of any contributing chunk; that chunk also provides the snippet.
    """
    groups: dict[tuple, list[tuple[int, RetrievedChunk]]] = {}
    for rank, rc in enumerate(retrieved):
        cit = rc.chunk.citation
        key: tuple
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
    for rank, rc in sorted(lined, key=lambda m: m[1].chunk.citation.line_start):
        cit = rc.chunk.citation
        if spans and cit.line_start <= spans[-1]["end"] + 1:
            cur = spans[-1]
            cur["end"] = max(cur["end"], cit.line_end)
            if rank < cur["rank"]:
                cur["rank"], cur["rc"] = rank, rc
        else:
            spans.append({"start": cit.line_start, "end": cit.line_end,
                          "rank": rank, "rc": rc})

    out: list[tuple[int, SourceView]] = []
    for span in spans:
        cit = span["rc"].chunk.citation
        merged_cit = replace(cit, line_start=span["start"], line_end=span["end"])
        out.append((span["rank"], SourceView(
            title=cit.title,
            url=github_url(merged_cit),
            range_label=f"lines {span['start']}–{span['end']}",
            meta_label=f"{cit.repo} @ {(cit.commit_hash or '')[:7]}",
            snippet=span["rc"].chunk.text[:SNIPPET_CHARS],
        )))
    for rank, rc in unlined:  # defensive: chunker always sets line spans on GitHub chunks
        cit = rc.chunk.citation
        out.append((rank, SourceView(
            title=cit.title, url=github_url(cit), range_label="",
            meta_label=f"{cit.repo} @ {(cit.commit_hash or '')[:7]}",
            snippet=rc.chunk.text[:SNIPPET_CHARS],
        )))
    return out


def _file_view(members: list[tuple[int, RetrievedChunk]]) -> tuple[int, SourceView]:
    rank, rc = min(members, key=lambda m: m[0])
    cit = rc.chunk.citation
    return (rank, SourceView(
        title=cit.title,
        url=f"/documents/{cit.doc_id}/file",
        range_label=cit.page_or_section or "",
        meta_label="uploaded document",
        snippet=rc.chunk.text[:SNIPPET_CHARS],
    ))
```

Note: deleted documents are removed from the search corpus on delete (see
`delete_document` in `web/app.py`), so a tombstoned doc effectively never appears in
`retrieved`; the `/documents/{doc_id}/file` route's 404 (Task 8) is the backstop.

- [x] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/core/test_sources.py -v`
Expected: 9 PASS

- [x] **Step 5: Lint and commit**

```bash
uv run ruff check src/genacademy_rag/core/sources.py tests/core/test_sources.py
git add src/genacademy_rag/core/sources.py tests/core/test_sources.py
git commit -m "feat: merge retrieved citations into deduped SourceView rows"
```

---

## Task 4: Overview-format answer prompt

**Files:**
- Modify: `src/genacademy_rag/core/graph.py:14-18` (ANSWER_SYSTEM) and `:40` (max_tokens)
- Modify: `tests/core/test_graph.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/core/test_graph.py`:

```python
def test_answer_prompt_keeps_grounding_and_adds_overview_format():
    from genacademy_rag.core.graph import ANSWER_SYSTEM

    # grounding rules must survive the format change verbatim
    assert "You answer ONLY from the provided course context" in ANSWER_SYSTEM
    assert "Never use outside knowledge" in ANSWER_SYSTEM
    # new output shape
    assert "overview paragraph" in ANSWER_SYSTEM
    assert "Be concise" not in ANSWER_SYSTEM


def test_answer_node_generates_with_800_max_tokens(fake_provider):
    from genacademy_rag.core.graph import build_graph
    from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk

    captured = {}
    real_generate = fake_provider.generate

    def recording_generate(messages, *, json_mode=False, max_tokens=512, temperature=0.0):
        if not json_mode:  # the answer call (the grader call uses json_mode=True)
            captured["max_tokens"] = max_tokens
        return real_generate(messages, json_mode=json_mode,
                             max_tokens=max_tokens, temperature=temperature)

    fake_provider.generate = recording_generate

    cit = Citation(doc_id="d1", title="README.md", source_type="github", repo="r",
                   file_path="README.md", commit_hash="abc", line_start=1, line_end=2)
    chunk = Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0,
                  text="RAG retrieves then generates.", citation=cit)

    class _Retriever:
        def retrieve(self, q):
            return [RetrievedChunk(chunk=chunk, score=0.9)]

    graph = build_graph(retriever=_Retriever(), provider=fake_provider)
    out = graph.invoke({"question": "What is RAG?"})
    assert out["refused"] is False
    assert captured["max_tokens"] == 800
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/core/test_graph.py -v -k "overview or 800"`
Expected: both FAIL (old prompt says "Be concise"; max_tokens is 512)

- [x] **Step 3: Implement**

In `src/genacademy_rag/core/graph.py`, replace the `ANSWER_SYSTEM` constant:

```python
ANSWER_SYSTEM = (
    "You answer ONLY from the provided course context. If the context does not contain the "
    "answer, say you could not find it. Never use outside knowledge. "
    "Format your answer as a short overview paragraph (2-3 sentences) that directly answers "
    "the question, followed by 2-4 key-point bullet lines starting with '- ' when the context "
    "supports them. Do not pad with information that is not in the context."
)
```

And in `answer_node`, change the generate call's `max_tokens`:

```python
        answer = provider.generate(
            [{"role": "system", "content": ANSWER_SYSTEM},
             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['question']}"}],
            json_mode=False, max_tokens=800,
        )
```

- [x] **Step 4: Run the full graph test file**

Run: `uv run pytest tests/core/test_graph.py -v`
Expected: all PASS

- [x] **Step 5: Commit**

```bash
git add src/genacademy_rag/core/graph.py tests/core/test_graph.py
git commit -m "feat: overview-format answer prompt, max_tokens 800"
```

---

## Task 5: `QueryResult.sources`

**Files:**
- Modify: `src/genacademy_rag/core/pipeline.py:79-105`
- Modify: `tests/core/test_query_pipeline.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/core/test_query_pipeline.py` (reuse the file's existing retriever/provider fixtures if equivalent ones exist; otherwise these are self-contained):

```python
def test_answer_result_carries_merged_sources(fake_provider):
    from genacademy_rag.core.pipeline import QueryPipeline
    from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk

    cit = Citation(doc_id="d1", title="README.md", source_type="github",
                   repo="awesome-agentic-ai-resources", file_path="README.md",
                   commit_hash="5dfb8691180dc4956107e86839998ba3a2ebd94f",
                   line_start=41, line_end=57)
    cit2 = Citation(doc_id="d1", title="README.md", source_type="github",
                    repo="awesome-agentic-ai-resources", file_path="README.md",
                    commit_hash="5dfb8691180dc4956107e86839998ba3a2ebd94f",
                    line_start=57, line_end=70)

    class _Retriever:
        def retrieve(self, q):
            return [
                RetrievedChunk(chunk=Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0,
                                           text="chunk one", citation=cit), score=0.9),
                RetrievedChunk(chunk=Chunk(chunk_id="d1::1", doc_id="d1", ordinal=1,
                                           text="chunk two", citation=cit2), score=0.8),
            ]

    qp = QueryPipeline(retriever=_Retriever(), provider=fake_provider)
    result = qp.answer("What did the course say about chunking?")
    assert result.refused is False
    # citations stay raw and per-chunk for the eval — unchanged contract
    assert len(result.citations) == 2
    # sources are the merged presentation rows
    assert len(result.sources) == 1
    assert result.sources[0].range_label == "lines 41–70"


def test_refused_result_has_empty_sources():
    from genacademy_rag.core.pipeline import QueryPipeline
    from tests.conftest import FakeModelProvider

    refusing = FakeModelProvider(canned_json='{"answerable": false, "confidence": 1}')

    class _EmptyRetriever:
        def retrieve(self, q):
            return []

    qp = QueryPipeline(retriever=_EmptyRetriever(), provider=refusing)
    result = qp.answer("Who won the 2030 World Cup?")
    assert result.refused is True
    assert result.sources == []
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/core/test_query_pipeline.py -v -k sources`
Expected: FAIL — `QueryResult` has no attribute `sources`

- [x] **Step 3: Implement**

In `src/genacademy_rag/core/pipeline.py`: add imports and extend `QueryResult` + `answer()`:

```python
from dataclasses import dataclass, field

from genacademy_rag.core.sources import SourceView, merge_citations
```

```python
@dataclass(frozen=True)
class QueryResult:
    answer: str
    citations: list[Citation]
    refused: bool
    confidence: int
    used_fallback: bool = False
    # Merged presentation rows for the answer card. Empty on refusals (the refused card
    # shows no source list). `citations` above stays raw/per-chunk — the eval contract.
    sources: list[SourceView] = field(default_factory=list)
```

```python
    def answer(self, question: str) -> QueryResult:
        out = self._graph.invoke({"question": question})
        # Index required keys directly (never `.get(default)`): the graph always sets answer,
        # citations, refused, confidence, used_fallback, so a missing key is a wiring bug we want
        # to surface as a KeyError — not paper over with an uncited / zero-confidence answer.
        return QueryResult(
            answer=out["answer"],
            citations=out["citations"],
            refused=out["refused"],
            confidence=out["confidence"],
            used_fallback=out["used_fallback"],
            sources=[] if out["refused"] else merge_citations(out["retrieved"]),
        )
```

- [x] **Step 4: Run to verify everything passes**

Run: `uv run pytest tests/core/test_query_pipeline.py tests/eval -v`
Expected: all PASS (eval tests prove the `citations` contract is untouched)

- [x] **Step 5: Commit**

```bash
git add src/genacademy_rag/core/pipeline.py tests/core/test_query_pipeline.py
git commit -m "feat: QueryResult carries merged SourceView rows"
```

---

## Task 6: Datastore — feedback table, `log_query` returns id

**Files:**
- Modify: `src/genacademy_rag/data/datastore.py` (SCHEMA `:20-49`, `log_query` `:328-353`, new methods after `recent_usage`)
- Modify: `tests/data/test_datastore.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/data/test_datastore.py` (the file's existing tests construct `SQLiteDatastore(tmp_path / "test.db")` — follow that pattern):

```python
def _log_one(ds, email="member@genacademy.local"):
    return ds.log_query(user_email=email, question="q?", refused=False, confidence=4,
                        used_fallback=False, n_citations=2, latency_ms=120)


def test_log_query_returns_row_id(tmp_path):
    from genacademy_rag.data.datastore import SQLiteDatastore
    ds = SQLiteDatastore(tmp_path / "t.db")
    first = _log_one(ds)
    second = _log_one(ds)
    assert isinstance(first, int)
    assert second == first + 1


def test_feedback_insert_and_summary(tmp_path):
    from genacademy_rag.data.datastore import SQLiteDatastore
    ds = SQLiteDatastore(tmp_path / "t.db")
    qid = _log_one(ds)
    ds.add_feedback(usage_log_id=qid, user_email="a@x.com", verdict=1)
    ds.add_feedback(usage_log_id=qid, user_email="b@x.com", verdict=-1)
    assert ds.feedback_summary() == {"up": 1, "down": 1}


def test_feedback_upsert_flips_verdict_not_duplicates(tmp_path):
    from genacademy_rag.data.datastore import SQLiteDatastore
    ds = SQLiteDatastore(tmp_path / "t.db")
    qid = _log_one(ds)
    ds.add_feedback(usage_log_id=qid, user_email="a@x.com", verdict=1)
    ds.add_feedback(usage_log_id=qid, user_email="a@x.com", verdict=-1)
    assert ds.feedback_summary() == {"up": 0, "down": 1}


def test_feedback_rejects_bad_verdict(tmp_path):
    import pytest
    from genacademy_rag.data.datastore import SQLiteDatastore
    ds = SQLiteDatastore(tmp_path / "t.db")
    qid = _log_one(ds)
    with pytest.raises(ValueError):
        ds.add_feedback(usage_log_id=qid, user_email="a@x.com", verdict=0)


def test_feedback_table_survives_reopen(tmp_path):
    """Migration-twice: re-opening the same DB must not error or lose feedback."""
    from genacademy_rag.data.datastore import SQLiteDatastore
    path = tmp_path / "t.db"
    ds = SQLiteDatastore(path)
    qid = _log_one(ds)
    ds.add_feedback(usage_log_id=qid, user_email="a@x.com", verdict=1)
    ds2 = SQLiteDatastore(path)
    assert ds2.feedback_summary() == {"up": 1, "down": 0}
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/data/test_datastore.py -v -k "feedback or returns_row_id"`
Expected: FAIL — `log_query` returns `None`; no `add_feedback` attribute

- [x] **Step 3: Implement**

In `SCHEMA` (append before the closing `"""`):

```sql
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    usage_log_id INTEGER NOT NULL REFERENCES usage_log(id),
    user_email TEXT NOT NULL,
    verdict INTEGER NOT NULL CHECK (verdict IN (-1, 1)),
    ts TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (usage_log_id, user_email));
```

(`CREATE TABLE IF NOT EXISTS` inside `SCHEMA` runs on every `__init__`, so existing DBs
gain the table on next open — same idempotent pattern as the other tables.)

Change `log_query`'s tail (return type `None` → `int`):

```python
    def log_query(
        self,
        *,
        user_email: str | None,
        question: str,
        refused: bool,
        confidence: int,
        used_fallback: bool,
        n_citations: int,
        latency_ms: int,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO usage_log(user_email, question, refused, confidence, used_fallback, "
                "n_citations, latency_ms) VALUES (?,?,?,?,?,?,?)",
                (
                    user_email,
                    question,
                    int(refused),
                    confidence,
                    int(used_fallback),
                    n_citations,
                    latency_ms,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)
```

Add after `recent_usage`:

```python
    def add_feedback(self, *, usage_log_id: int, user_email: str, verdict: int) -> None:
        if verdict not in (-1, 1):
            raise ValueError("verdict must be -1 or 1")
        with self._lock:
            self._conn.execute(
                "INSERT INTO feedback(usage_log_id, user_email, verdict) VALUES (?,?,?) "
                "ON CONFLICT(usage_log_id, user_email) DO UPDATE SET "
                "verdict=excluded.verdict, ts=CURRENT_TIMESTAMP",
                (usage_log_id, user_email, verdict),
            )
            self._conn.commit()

    def feedback_summary(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(verdict=1),0) AS up, "
                "COALESCE(SUM(verdict=-1),0) AS down FROM feedback"
            ).fetchone()
            return {"up": row["up"], "down": row["down"]}
```

- [x] **Step 4: Run the whole datastore file**

Run: `uv run pytest tests/data/test_datastore.py -v`
Expected: all PASS

- [x] **Step 5: Commit**

```bash
git add src/genacademy_rag/data/datastore.py tests/data/test_datastore.py
git commit -m "feat: feedback table + log_query returns row id"
```

---

## Task 7: `POST /feedback` endpoint

**Files:**
- Modify: `src/genacademy_rag/web/app.py` (new route after `ask`, `:180`)
- Modify: `tests/web/test_app.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/web/test_app.py` (use the file's existing `_client` helper and its login pattern — existing tests log in by posting `/login` with the seeded `member@genacademy.local`/`member` credentials after fetching the CSRF token from the login page; follow the same regex extraction used there):

```python
def _login_member(client):
    page = client.get("/login").text
    token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
    client.post("/login", data={"email": "member@genacademy.local",
                                "password": "member", "csrf_token": token})
    return token


def _ask_and_get_query_id(client, token):
    page = client.post("/ask", data={"question": "What is RAG?",
                                     "csrf_token": token}).text
    m = re.search(r'name="query_id" value="(\d+)"', page)
    return m.group(1) if m else None, page


def test_feedback_requires_login(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    r = client.post("/feedback", data={"query_id": 1, "verdict": 1, "csrf_token": "x"},
                    follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_feedback_requires_csrf(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _login_member(client)
    r = client.post("/feedback", data={"query_id": 1, "verdict": 1, "csrf_token": "wrong"})
    assert r.status_code == 403


def test_feedback_rejects_bad_verdict(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    token = _login_member(client)
    r = client.post("/feedback", data={"query_id": 1, "verdict": 7, "csrf_token": token})
    assert r.status_code == 400


def test_feedback_happy_path_persists_and_returns_fragment(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    token = _login_member(client)
    qid, _ = _ask_and_get_query_id(client, token)
    assert qid is not None
    r = client.post("/feedback", data={"query_id": qid, "verdict": 1, "csrf_token": token})
    assert r.status_code == 200
    assert "Thanks for the feedback" in r.text
    ds = client.app.state.datastore
    assert ds.feedback_summary()["up"] == 1


def test_feedback_write_failure_does_not_500(monkeypatch, tmp_path, caplog):
    client = _client(monkeypatch, tmp_path)
    token = _login_member(client)
    ds = client.app.state.datastore

    def boom(**kwargs):
        raise RuntimeError("db down")

    ds.add_feedback = boom
    with caplog.at_level(logging.ERROR):
        r = client.post("/feedback", data={"query_id": 1, "verdict": 1, "csrf_token": token})
    assert r.status_code == 200  # best-effort: user never sees a 500
    assert "feedback write failed" in caplog.text
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/web/test_app.py -v -k feedback`
Expected: FAIL — 404/405 (route does not exist)

- [x] **Step 3: Implement**

In `src/genacademy_rag/web/app.py`, after the `ask` route:

```python
    @app.post("/feedback", response_class=HTMLResponse)
    def feedback(
        request: Request,
        query_id: int = Form(...),
        verdict: int = Form(...),
        csrf_token_value: str | None = Form(None, alias="csrf_token"),
    ):
        user = current_user(request)
        if not user:
            return RedirectResponse("/login", status_code=303)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        if verdict not in (1, -1):
            return HTMLResponse("Bad verdict", status_code=400)
        try:
            datastore.add_feedback(usage_log_id=query_id, user_email=user, verdict=verdict)
        except Exception:
            # Best-effort, like log_query: feedback must never break the answer view.
            logger.exception("feedback write failed (query_id=%r)", query_id)
        return HTMLResponse('<span class="text-xs text-slate-500">Thanks for the feedback</span>')
```

Note: the happy-path test also needs Task 9's template change (the hidden `query_id`
field). If executing tasks in order, mark `test_feedback_happy_path_persists_and_returns_fragment`
with `@pytest.mark.xfail(reason="query_id field lands with the chat.html rebuild", strict=True)`
now and remove the marker in Task 9 — or write the test in Task 9 instead. The other four
tests must pass now.

- [x] **Step 4: Run to verify**

Run: `uv run pytest tests/web/test_app.py -v -k feedback`
Expected: 5 PASS, 1 xfail (happy path until Task 9)

- [x] **Step 5: Commit**

```bash
git add src/genacademy_rag/web/app.py tests/web/test_app.py
git commit -m "feat: POST /feedback endpoint (CSRF, best-effort persist)"
```

---

## Task 8: `GET /documents/{doc_id}/file`

**Files:**
- Modify: `src/genacademy_rag/web/app.py` (import `FileResponse`; new route after `feedback`)
- Modify: `tests/web/test_app.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/web/test_app.py`:

```python
def _store_doc(client, tmp_path, *, doc_id="up1", suffix=".pdf", status="indexed"):
    stored = tmp_path / f"deck{suffix}"
    stored.write_bytes(b"%PDF-1.4 fake content")
    ds = client.app.state.datastore
    ds.add_document(id=doc_id, title="Week2 Deck", source_type=suffix.lstrip("."),
                    filename=f"deck{suffix}", uploaded_by="admin@genacademy.local",
                    status=status, stored_path=str(stored))
    return stored


def test_document_file_requires_login(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    r = client.get("/documents/up1/file", follow_redirects=False)
    assert r.status_code == 303


def test_document_file_serves_pdf_inline(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _login_member(client)
    _store_doc(client, tmp_path, suffix=".pdf")
    r = client.get("/documents/up1/file")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert "inline" in r.headers["content-disposition"]


def test_document_file_downloads_pptx(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _login_member(client)
    _store_doc(client, tmp_path, doc_id="up2", suffix=".pptx")
    r = client.get("/documents/up2/file")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]


def test_document_file_404s(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _login_member(client)
    # unknown doc
    assert client.get("/documents/nope/file").status_code == 404
    # tombstoned doc
    _store_doc(client, tmp_path, doc_id="dead", status="deleted")
    assert client.get("/documents/dead/file").status_code == 404
    # GitHub-sourced doc (no stored_path)
    ds = client.app.state.datastore
    ds.add_document(id="gh1", title="README.md", source_type="github",
                    repo="r", file_path="README.md", commit_hash="abc")
    assert client.get("/documents/gh1/file").status_code == 404
```

(If `add_document`'s signature differs from these kwargs, match the call sites in
`scripts/ingest_eval_corpus.py` / the upload route — it is keyword-based `**kwargs`
mapped to the `documents` columns shown in `SCHEMA`.)

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/web/test_app.py -v -k document_file`
Expected: FAIL — 404 on every call including happy path (route missing)

- [x] **Step 3: Implement**

In `src/genacademy_rag/web/app.py`, extend the responses import:

```python
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
```

Add the route:

```python
    @app.get("/documents/{doc_id}/file")
    def document_file(request: Request, doc_id: str):
        # Members may download source documents: they already see this content as
        # retrieved chunks, so serving the original leaks nothing new.
        if not current_user(request):
            return RedirectResponse("/login", status_code=303)
        doc = datastore.get_document(doc_id)
        if not doc or doc.get("status") == "deleted" or not doc.get("stored_path"):
            return HTMLResponse("Not found", status_code=404)
        path = Path(doc["stored_path"])
        if not path.exists():
            return HTMLResponse("Not found", status_code=404)
        filename = doc.get("filename") or path.name
        if path.suffix.lower() == ".pdf":
            return FileResponse(path, media_type="application/pdf",
                                filename=filename, content_disposition_type="inline")
        return FileResponse(path, filename=filename)  # attachment by default
```

The path always comes from the datastore row — never from the request — so there is no
traversal surface.

- [x] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/web/test_app.py -v -k document_file`
Expected: 4 PASS

- [x] **Step 5: Commit**

```bash
git add src/genacademy_rag/web/app.py tests/web/test_app.py
git commit -m "feat: serve stored uploads at /documents/{doc_id}/file"
```

---

## Task 9: Rebuild `chat.html` + thread `query_id`/`confidence_bucket` through `/ask`

**Files:**
- Modify: `src/genacademy_rag/web/templates/chat.html` (full rewrite)
- Modify: `src/genacademy_rag/web/app.py` (`ask` view `:153-180`, import)
- Modify: `tests/web/test_app.py`

- [x] **Step 1: Write the failing tests**

Append to `tests/web/test_app.py` (and remove the Task-7 `xfail` marker from the
feedback happy-path test):

```python
def test_answer_card_renders_badge_sources_disclaimer(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    token = _login_member(client)
    _, page = _ask_and_get_query_id(client, token)
    # confidence badge (canned grader JSON says confidence 5 -> high)
    assert "High confidence" in page
    # clickable pinned-commit source link (the _client retriever cites repo r @ abc123)
    assert 'href="https://github.com/The-Gen-Academy/r/blob/abc123/README.md#L1-L2"' in page
    # snippet from the retrieved chunk
    assert "RAG retrieves then generates." in page
    # disclaimer
    assert "it can make mistakes" in page
    # copy + retry affordances
    assert "copy" in page and "retry" in page


def test_refused_card_has_refusal_badge_no_copy_no_sources(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, refused=True)
    token = _login_member(client)
    _, page = _ask_and_get_query_id(client, token)
    assert "Not in the materials" in page
    assert "Sources" not in page
    assert "⧉ copy" not in page
    assert "it can make mistakes" in page


def test_thumbs_hidden_when_log_query_fails(monkeypatch, tmp_path, caplog):
    client = _client(monkeypatch, tmp_path)
    token = _login_member(client)
    ds = client.app.state.datastore

    def boom(**kwargs):
        raise RuntimeError("db down")

    ds.log_query = boom
    with caplog.at_level(logging.ERROR):
        qid, page = _ask_and_get_query_id(client, token)
    assert qid is None          # no hidden query_id field
    assert "👍" not in page     # thumbs not rendered
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/web/test_app.py -v -k "answer_card or refused_card or thumbs_hidden or feedback_happy"`
Expected: FAIL (old template has none of these elements)

- [x] **Step 3: Update the `ask` view**

In `src/genacademy_rag/web/app.py` add the import:

```python
from genacademy_rag.core.sources import confidence_bucket
```

Replace the `ask` route body after `result = qp.answer(question)`:

```python
        latency_ms = int((time.perf_counter() - start) * 1000)
        query_id = None
        try:
            query_id = datastore.log_query(
                user_email=current_user(request),
                question=question,
                refused=result.refused,
                confidence=result.confidence,
                used_fallback=result.used_fallback,
                n_citations=len(result.citations),
                latency_ms=latency_ms,
            )
        except Exception:
            logger.exception("usage log_query failed (question=%r)", question)
        return TEMPLATES.TemplateResponse(
            request,
            "chat.html",
            csrf_context(request, {
                "result": result,
                "question": question,
                "query_id": query_id,
                "bucket": None if result.refused else confidence_bucket(result.confidence),
            }),
        )
```

Also update the `home` route's context so the template's new variables are always
defined:

```python
        return TEMPLATES.TemplateResponse(
            request, "chat.html",
            csrf_context(request, {"result": None, "question": None,
                                   "query_id": None, "bucket": None}),
        )
```

- [x] **Step 4: Rewrite the template**

Replace `src/genacademy_rag/web/templates/chat.html` entirely:

```html
<!doctype html><html><head><meta charset="utf-8"><title>GenAcademy RAG</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/htmx.org@1.9.12"></script>
<script defer src="https://unpkg.com/alpinejs@3.14.1/dist/cdn.min.js"></script>
<style>[x-cloak]{display:none}</style></head>
<body class="bg-slate-50 min-h-screen">
<div class="max-w-2xl mx-auto p-6 space-y-4">
  <h1 class="text-2xl font-semibold">Ask the cohort materials</h1>
  <form method="post" action="/ask" class="flex gap-2">
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
    <input name="question" value="{{ question or '' }}" placeholder="What did the course say about…?"
           class="flex-1 border rounded px-3 py-2">
    <button class="bg-slate-900 text-white rounded px-4">Ask</button>
  </form>

  {% if result %}
  <div class="bg-white rounded-xl shadow p-5 space-y-3" x-data="{copied:false}">

    <div class="flex items-center justify-between">
      {% if result.refused %}
        <span class="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-800">● Not in the materials</span>
      {% else %}
        <span class="text-xs font-semibold px-2.5 py-0.5 rounded-full
                     {% if bucket == 'high' %}bg-green-100 text-green-800{% elif bucket == 'medium' %}bg-amber-100 text-amber-800{% else %}bg-slate-200 text-slate-700{% endif %}"
              title="Reflects the grader's answerability signal ({{ result.confidence }}/5), not a fact-checked probability">
          ● {{ bucket|capitalize }} confidence</span>
      {% endif %}

      <div class="flex items-center gap-3 text-sm text-slate-500">
        {% if not result.refused %}
        <button type="button" title="Copy answer"
                @click="navigator.clipboard.writeText($refs.answer.innerText); copied = true; setTimeout(() => copied = false, 1500)">
          <span x-show="!copied">⧉ copy</span><span x-show="copied" x-cloak>✓ copied</span>
        </button>
        {% endif %}
        <form method="post" action="/ask" class="inline">
          <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
          <input type="hidden" name="question" value="{{ question }}">
          <button title="Retry this question">↻ retry</button>
        </form>
        {% if query_id %}
        <div id="feedback-box" class="flex gap-2">
          <form hx-post="/feedback" hx-target="#feedback-box" hx-swap="outerHTML" class="inline">
            <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
            <input type="hidden" name="query_id" value="{{ query_id }}">
            <input type="hidden" name="verdict" value="1">
            <button title="Good answer">👍</button>
          </form>
          <form hx-post="/feedback" hx-target="#feedback-box" hx-swap="outerHTML" class="inline">
            <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
            <input type="hidden" name="query_id" value="{{ query_id }}">
            <input type="hidden" name="verdict" value="-1">
            <button title="Bad answer">👎</button>
          </form>
        </div>
        {% endif %}
      </div>
    </div>

    {% if result.refused %}
      <p class="text-amber-700 font-medium">{{ result.answer }}</p>
    {% else %}
      <p class="whitespace-pre-wrap leading-relaxed" x-ref="answer">{{ result.answer }}</p>

      {% if result.sources %}
      <div class="pt-2 border-t">
        <p class="text-xs uppercase tracking-wide text-slate-500 mb-1">Sources ({{ result.sources|length }})</p>
        {% for s in result.sources %}
        <details class="text-sm mb-1" {% if loop.first %}open{% endif %}>
          <summary class="cursor-pointer">
            {% if s.url %}
              <a href="{{ s.url }}" target="_blank" rel="noopener" class="text-blue-700 underline">{{ s.title }}{% if s.range_label %} · {{ s.range_label }}{% endif %} ↗</a>
            {% else %}
              {{ s.title }}{% if s.range_label %} · {{ s.range_label }}{% endif %}
            {% endif %}
            <span class="text-slate-500 text-xs">— {{ s.meta_label }}</span>
          </summary>
          <div class="mt-1 ml-4 p-2 bg-slate-50 border-l-2 border-slate-300 text-slate-600 font-mono text-xs whitespace-pre-wrap">{{ s.snippet }}</div>
        </details>
        {% endfor %}
      </div>
      {% endif %}
    {% endif %}

    <p class="text-xs text-slate-400 pt-1">AI-generated from course materials — it can make mistakes. Check the sources above.</p>
  </div>
  {% endif %}
</div></body></html>
```

- [x] **Step 5: Run the full web suite**

Run: `uv run pytest tests/web -v`
Expected: all PASS (including the un-xfailed feedback happy path)

- [x] **Step 6: Commit**

```bash
git add src/genacademy_rag/web/templates/chat.html src/genacademy_rag/web/app.py tests/web/test_app.py
git commit -m "feat: answer card with badge, sources, copy/retry, thumbs, disclaimer"
```

---

## Task 10: Admin dashboard feedback counts

**Files:**
- Modify: `src/genacademy_rag/web/app.py` (`admin_dashboard` `:284-296`)
- Modify: `src/genacademy_rag/web/templates/admin_dashboard.html`
- Modify: `tests/web/test_app.py`

- [x] **Step 1: Write the failing test**

Append to `tests/web/test_app.py` (existing admin tests log in as
`admin@genacademy.local`/`admin` — reuse that pattern):

```python
def test_dashboard_shows_feedback_counts(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    page = client.get("/login").text
    token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
    client.post("/login", data={"email": "admin@genacademy.local",
                                "password": "admin", "csrf_token": token})
    ds = client.app.state.datastore
    qid = ds.log_query(user_email="m@x.com", question="q?", refused=False, confidence=4,
                       used_fallback=False, n_citations=1, latency_ms=50)
    ds.add_feedback(usage_log_id=qid, user_email="m@x.com", verdict=1)
    page = client.get("/admin/dashboard").text
    assert "Thumbs up" in page and "Thumbs down" in page
```

- [x] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/web/test_app.py -v -k dashboard_shows_feedback`
Expected: FAIL — "Thumbs up" not in page

- [x] **Step 3: Implement**

In `admin_dashboard`, add feedback to the context:

```python
        return TEMPLATES.TemplateResponse(
            request,
            "admin_dashboard.html",
            csrf_context(request, {"summary": summary, "rows": rows,
                                   "feedback": datastore.feedback_summary()}),
        )
```

In `admin_dashboard.html`, widen the stats grid and add two tiles — change the section
opening tag to `md:grid-cols-7` and append the tiles inside it:

```html
  <section class="grid grid-cols-2 md:grid-cols-7 gap-3">
    ...existing five tiles unchanged...
    <div class="bg-white rounded shadow p-3"><p class="text-xs text-slate-500">Thumbs up</p><p class="text-2xl">👍 {{ feedback.up }}</p></div>
    <div class="bg-white rounded shadow p-3"><p class="text-xs text-slate-500">Thumbs down</p><p class="text-2xl">👎 {{ feedback.down }}</p></div>
  </section>
```

- [x] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/web -v`
Expected: all PASS

- [x] **Step 5: Commit**

```bash
git add src/genacademy_rag/web/app.py src/genacademy_rag/web/templates/admin_dashboard.html tests/web/test_app.py
git commit -m "feat: feedback counts on admin dashboard"
```

---

## Task 11: Full gates + faithfulness eval re-run

**Files:**
- Modify: `eval/REPORT.md` (regenerated)

- [ ] **Step 1: Full lint + test gate**

```bash
uv run ruff check .
uv run pytest
```
Expected: ruff clean; all tests pass (none skipped except `@pytest.mark.integration` without keys).

- [ ] **Step 2: Re-run the eval on the Nebius preset**

The `ANSWER_SYSTEM` change affects generated answers, so the faithfulness score must be
re-measured (spec §5; roadmap: any regenerated report runs on Nebius). Requires
`NEBIUS_API_KEY` in the environment:

```bash
GENACADEMY_PROVIDER=nebius uv run python scripts/run_eval.py
```

Expected: `eval/REPORT.md` regenerated. Retrieval metrics (recall/precision/MRR) must be
**identical** to the previous run — nothing in this slice touches retrieval; any drift
is a regression, stop and investigate.

- [ ] **Step 3: Record the before/after faithfulness delta**

In `eval/REPORT.md` (or its changelog section if the generator preserves one), note:
faithfulness before (58%, terse-answer prompt) vs after (new overview prompt), judge
model used, and that retrieval metrics are unchanged by construction.

- [ ] **Step 4: Commit**

```bash
git add eval/REPORT.md
git commit -m "eval: re-measure faithfulness after overview-format prompt"
```

---

## Task 12: Decisions doc + PR

**Files:**
- Create: `docs/answer-ux-decisions-and-tradeoffs.md`

- [ ] **Step 1: Write the decisions doc**

Create `docs/answer-ux-decisions-and-tradeoffs.md`:

```markdown
# Answer Trust & Feedback UX — Decisions and Tradeoffs

Companion to `superpowers/specs/2026-06-10-answer-trust-feedback-ux-design.md`
(same series as the phase 0/1/2 decision docs).

## Confidence is a bucket, not a percentage
The grader's 1-5 confidence is self-reported by an LLM (or derived from a cosine
threshold on the fallback path). Showing "80%" would imply calibration that does not
exist. Low/Med/High + a tooltip stating the basis is the honest rendering.

## Citations merge at presentation time, not in the pipeline
The eval scores per-chunk citations; merging overlapping line ranges in `core/` data
would change the graded contract. So `QueryResult.citations` stays raw and a separate
`sources` field carries the merged presentation rows — two consumers, two shapes, one
source of truth (`retrieved`).

## Snippets come from `retrieved`, not a second lookup
The graph state already carries every retrieved chunk's text. Re-fetching chunk text
from the datastore at render time would add a query per citation and a second
consistency domain. Zero new plumbing: `merge_citations` reads what the graph returns.

## Feedback is an upsert keyed on (query, user)
Re-clicking flips a verdict instead of stuffing the table. Feedback writes are
best-effort: a DB failure logs an error and the user still gets their answer —
same posture as `log_query`.

## HTMX fragment swap instead of PRG for thumbs
Answers are not addressable URLs (stateless form-post), so a 303 redirect after
feedback would land on `/` and erase the rendered answer. The thumbs POST via HTMX
and swap in a "thanks" fragment; the answer stays on screen.

## Uploaded sources link to a served file, not nothing
Phase 1 already persists upload bytes for re-indexing (`stored_path`), so
`/documents/{doc_id}/file` makes uploaded PDF/PPTX/DOCX sources one-click verifiable
too — the same trust property as the pinned-commit GitHub links. Login-gated; the
path always comes from the datastore row, never the request.

## Overview answers raise the faithfulness stakes
A longer answer has more room to hallucinate. The grounding instructions are
unchanged and the faithfulness eval is re-run with the before/after delta recorded —
"measured, not asserted" applies to prompt changes too.
```

- [ ] **Step 2: Commit and open the PR**

```bash
git add docs/answer-ux-decisions-and-tradeoffs.md
git commit -m "docs: answer-UX decisions and tradeoffs"
git push -u origin feat/answer-trust-feedback-ux
gh pr create --title "Answer trust & feedback UX" --body "$(cat <<'EOF'
Implements docs/superpowers/specs/2026-06-10-answer-trust-feedback-ux-design.md:
clickable pinned-commit + uploaded-document source links, merged citation ranges with
snippets, overview-format answers (faithfulness re-measured), Low/Med/High confidence
badge, persisted thumbs up/down with admin counts, copy/retry, AI-mistake disclaimer.

Deterministic retrieval eval untouched (verified identical in the regenerated report).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Builder ≠ reviewer gate**

Per `AGENTS.md`, a different model / fresh context must review the PR before merge
(the project's standard `pr-review-toolkit` pass). Do not merge without it.

---

## Forward pointers (deliberately not in this plan)

- HTMX fragment swap for `/ask` itself (no full-page reload) and streaming answers.
- Feedback-driven eval mining (turn 👎-ed questions into gold-set candidates).
- Markdown rendering of answer bullets (currently rendered as pre-wrapped plain text —
  fine for "- " bullets; revisit only if answers grow richer structure).
