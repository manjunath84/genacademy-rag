# Answer Trust & Feedback UX — Design

**Date:** 2026-06-10 · **Status:** shipped in PR #11
**Scope:** one pre-submission slice. Brainstormed with visual mockups (`.superpowers/brainstorm/`);
the approved card layout is `answer-card-v2.html` in that session.

## 1. Problem

The current answer card (`web/templates/chat.html`) undermines trust in an otherwise
citation-disciplined pipeline:

- Source rows are not clickable — the reader cannot verify a citation without manually
  navigating GitHub.
- The answer node cites **every** retrieved chunk verbatim, so one answer shows five
  overlapping line-ranges of the same README.
- Answers are terse single sentences (`ANSWER_SYSTEM` says "Be concise"), below the
  overview quality users expect from an AI answer surface.
- The pipeline computes a 1–5 confidence bucket and a `used_fallback` flag, but neither
  is shown to the user.
- No copy button, no retry, no feedback capture, no AI-mistake disclaimer.

## 2. Decisions (settled during brainstorming)

| Question | Decision |
|---|---|
| Timing | **Before the Week-2 submission** — tight scope, demo-visible items first |
| Feedback depth | **Persist + admin surface** — new table, `/feedback` endpoint, dashboard counts |
| Confidence display | **Low/Med/High badge** (1–2/3/4–5), tooltip says it is the grader's answerability signal, not a calibrated probability |
| Source row | **Clickable link + ~2–3 line snippet** of the retrieved chunk (option A of three mockups) |
| Reload button | **Plain retry at temperature 0** — transient-error recovery; no temperature change |
| Answer format | **Overview paragraph + key-point bullets** (Google-AI-overview shape), grounded only in retrieved chunks |

## 3. Answer card (approved layout)

**Answered state:**
- Header row: confidence badge left (High = green, Medium = amber, Low = gray), toolbar
  right: copy ⧉, retry ↻, 👍, 👎.
- Answer: overview paragraph (2–3 sentences) then key-point bullets when the material
  supports them.
- "Sources (N)": merged citations as `<details>` rows. Summary line =
  `<a href={github_url}>title · lines a–b ↗</a> — repo @ shortsha`; expanded body =
  snippet (~240 chars, monospace, left-rule). Uploaded files (pdf/docx/pptx): title +
  page/section label, linked to `GET /documents/{doc_id}/file` (see §4.3) — PDFs render
  inline, PPTX/DOCX download. Covers future admin uploads automatically (same `doc_id`
  chain). Only when no stored file exists (e.g. tombstoned doc) does the row fall back
  to unlinked title + snippet.
- Footer disclaimer (muted): *"AI-generated from course materials — it can make
  mistakes. Check the sources above."*

**Refused state:**
- Amber badge "Not in the materials", refusal text, retry + thumbs (no copy button),
  same disclaimer. No source list.

## 4. Architecture

### 4.1 Core (`src/genacademy_rag/core/`) — pure, offline-testable

**`graph.py`** — `ANSWER_SYSTEM` rewritten for the overview format. Grounding sentences
kept verbatim ("answer ONLY from the provided course context… Never use outside
knowledge"); the only change is the output-shape instruction (overview paragraph +
bullets) replacing "Be concise". `max_tokens` 512 → 800. No node/edge changes.

**New `core/sources.py`** — three pure functions plus a small frozen view dataclass:

```python
@dataclass(frozen=True)
class SourceView:
    title: str
    url: str | None        # GitHub pinned-commit URL, /documents/{doc_id}/file for uploads, None only when neither exists (e.g. tombstoned)
    range_label: str       # "lines 41–70" | "page 3"
    meta_label: str        # "awesome-agentic-ai-resources @ 5dfb869" | "uploaded document"
    snippet: str           # ~240 chars of the first contributing chunk

def merge_citations(retrieved: list[RetrievedChunk]) -> list[SourceView]
def github_url(citation: Citation) -> str | None
def confidence_bucket(confidence: int) -> str   # "low" | "medium" | "high"
```

- `merge_citations` groups by `(repo, file_path, commit_hash)` (or `doc_id` for files),
  merges **overlapping or adjacent** line ranges (e.g. 41–57, 57–65, 64–68, 67–70 →
  41–70), orders by best retrieval rank, keeps the highest-ranked chunk's snippet.
  Non-contiguous ranges in the same file stay separate rows.
- `github_url` builds
  `https://github.com/The-Gen-Academy/{repo}/blob/{commit_hash}/{file_path}#L{a}-L{b}`
  (owner matches the ingest allowlist in `core/loaders/__init__.py`). Returns `None`
  when `repo`/`commit_hash` is absent (uploaded files).
- `confidence_bucket`: 1–2 → low, 3 → medium, 4–5 → high.

**`pipeline.py`** — `QueryResult` gains `sources: list[SourceView]` (default empty),
built in `answer()` from `out["retrieved"]` via `merge_citations` **only when not
refused** (the refused card shows no source list, so `sources` stays empty there). **The eval-facing
`citations` field is unchanged byte-for-byte** — the deterministic eval and the
faithfulness judge keep consuming exactly what they consume today.

### 4.2 Data (`src/genacademy_rag/data/`)

- `log_query(...)` returns the inserted `usage_log` row id (`int`).
- New table, added to the idempotent `_migrate()`:

```sql
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    usage_log_id INTEGER NOT NULL REFERENCES usage_log(id),
    user_email TEXT NOT NULL,
    verdict INTEGER NOT NULL CHECK (verdict IN (-1, 1)),
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (usage_log_id, user_email)
);
```

- `add_feedback(usage_log_id, user_email, verdict)` — upsert on the unique pair, so
  re-clicking flips the verdict instead of duplicating.
- `feedback_summary() -> dict` — up/down totals for the dashboard.
- Both take the existing datastore `RLock`; same patterns as current methods.

### 4.3 Web (`src/genacademy_rag/web/`)

- **`chat.html`** rebuilt to the approved card. Alpine.js added (locked stack) for the
  copy button (`navigator.clipboard.writeText`, answer text only). Retry re-posts the
  same form (same question, fresh CSRF). Thumbs are two HTMX POST forms to `/feedback`
  with hidden `query_id`, `verdict`, `csrf_token`; the response is a small HTML fragment
  ("Thanks for the feedback") swapped over the thumbs. (A PRG redirect would land on `/`
  and lose the rendered answer — answers are not addressable URLs, so fragment swap it is;
  HTMX is already loaded in `chat.html`.)
- **`POST /feedback`** — requires login + valid CSRF; validates `verdict ∈ {1, -1}` and
  `query_id` is an int; the usage row must exist and belong to the current user before feedback is
  accepted. Calls `add_feedback` **best-effort** for storage failures (log on failure, never 500 the
  user).
- **`/ask` view** — threads `log_query`'s returned id into the template context as
  `query_id`. If `log_query` failed (already best-effort), `query_id` is `None` and the
  template hides the thumbs.
- **Admin dashboard** — 👍/👎 totals from `feedback_summary()` next to existing stats.
- **`GET /documents/{doc_id}/file`** — requires login (members already see this content
  as chunks, so serving the original leaks nothing new). Looks the document up in the
  datastore by `doc_id` and streams `stored_path` — never serves caller-supplied paths.
  `Content-Disposition: inline` for PDF, attachment for PPTX/DOCX. 404 when the doc is
  unknown, tombstoned, or has no `stored_path` (GitHub-sourced docs use their GitHub
  link instead and never hit this route). `SourceView.url` for uploaded files is built
  by `merge_citations` as `/documents/{doc_id}/file`.

### 4.4 Error handling

- Feedback write failure: logged, user still gets their answer page. Never blocks.
- `query_id` absent → thumbs not rendered (graceful degradation).
- Snippets are plain Jinja output (autoescaped); truncated to ~240 chars at build time
  in `merge_citations`, not in the template.

## 5. Eval impact & verification

- **Deterministic retrieval eval (recall/precision/MRR): untouched.** No retrieval,
  chunking, or grader change in this slice.
- **Faithfulness eval re-run** after the `ANSWER_SYSTEM` change — answer format directly affects the
  LLM-judge score. Report regenerated with before/after noted in `eval/REPORT.md`; faithfulness
  stayed at 58%.
- New unit tests (offline, FakeModelProvider unchanged):
  - `merge_citations`: overlap merge, adjacency merge, non-contiguous separation,
    multi-file, uploaded-file grouping, snippet truncation.
  - `github_url`: github citation, uploaded file → None, line-anchor format.
  - `confidence_bucket`: boundary values 1/2/3/4/5.
  - Datastore: `log_query` returns id; feedback insert, upsert-flip, summary counts;
    migration runs twice cleanly.
  - Web: `/feedback` CSRF required, login required, bad verdict rejected, best-effort
    on datastore failure; thumbs hidden when `query_id` is None; chat template renders
    sources/badge/disclaimer.
  - `/documents/{doc_id}/file`: login required, serves stored upload with correct
    disposition, 404 on unknown/tombstoned/no-stored-path doc, path comes only from the
    datastore lookup.
- Standard gates: `ruff` + `pytest` green; builder ≠ reviewer review before merge.

## 6. Constitution updates

- **`specs/mission.md`** — add four product principles under answer UX:
  1. Every citation is verifiable in one click (pinned-commit links).
  2. Confidence is shown honestly as a bucket, never as a probability.
  3. User feedback is captured and visible to admins (future eval-mining input).
  4. An AI-mistake disclaimer is always visible on generated answers.
- **`specs/roadmap.md`** — add this slice as **"Answer trust & feedback UX"**
  (pre-submission, shipped).
- **`specs/tech-stack.md`** — unchanged; everything here uses the locked stack
  (Alpine.js was already in it).

## 7. Out of scope (deliberate)

- HTMX fragment swap / streaming answers (seam noted; add post-submission).
- Regenerate-at-temperature (>0) — user chose plain retry.
- Feedback-driven eval mining, per-question feedback analytics.
- Pinecone/JSON API/client-side rendering.
- Any retrieval/chunking/grader change (separate slices with their own eval deltas).
