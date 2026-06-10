# GenAcademy RAG - Phase 2 Section-Aware Chunking Design

**Date:** 2026-06-09
**Status:** draft design for independent review before implementation planning
**Builds on:** Phase 0/1 on `main`, PR #3 cross-encoder rerank merged
**Source of scope:** `specs/roadmap.md` Phase 2 and the failure analysis in `eval/REPORT.md`
**Companion context:** `docs/learnings.md`

---

## 1. Goal

Add one independently droppable Phase 2 depth slice: **section-aware chunking** behind the existing
`Chunker` seam, measured with a before/after deterministic retrieval-eval delta.

The success artifact is not "a new chunker exists." The success artifact is an honest comparison over
the same pinned raw eval documents and gold set:

- fixed-size baseline: current `eval` collection and `eval/REPORT.md` metrics
- section-aware candidate: newly ingested alternate eval collection from the same pinned raw docs
- same embedder, same retriever settings, same gold set, same scorer
- latency/chunk-count/context-size impact reported alongside retrieval metrics

This slice must not touch Pinecone, Nebius embeddings, deploy, or product UI except for optional
configuration wiring. It should remain removable without destabilizing Phase 0/1.

---

## 2. Why This Is The Next Slice

The current failure table points at chunking more than at generation:

- q5: compact markdown table row lost useful section context
- q7: prerequisite table split across fixed-size windows
- q8: Week 6 resource table truncated into incomplete context
- q9/q10: multi-document questions are hurt by top-k pressure and fragmented context

PR #3 showed rerank can recover some ranking failures, but it does not fix the root issue where a
single chunk no longer contains the section header plus the answer row. Section-aware chunking should
attack that failure mode directly.

---

## 3. Non-Goals

- Do not ingest or inspect the excluded Week 2 sample-solution repo.
- Do not replace the `Chunker` interface.
- Do not mutate the existing `eval` Chroma collection.
- Do not change the gold set to make the new chunker look better.
- Do not add semantic clustering, query rewriting, multi-hop retrieval, or adjacent-chunk stitching in
  this slice.
- Do not make rerank part of the primary chunking delta unless explicitly reported as a secondary
  combined run.

---

## 4. Core Design

Add a new pure-core chunker:

```python
class SectionAwareChunker:
    def __init__(self, max_chars: int = 1500, overlap: int = 150): ...
    def chunk(self, doc: Document) -> list[Chunk]: ...
```

The implementation should preserve the existing `Chunker` Protocol:

```python
class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...
```

### Markdown / GitHub Text

For markdown-like documents, split on structural boundaries before falling back to character windows:

1. ATX headings (`#`, `##`, `###`, etc.)
2. fenced code blocks as indivisible blocks where possible
3. markdown table blocks as indivisible blocks where possible
4. paragraph/list blocks separated by blank lines
5. fixed-size fallback only when a single structural block exceeds `max_chars`

Each produced chunk should include its nearest heading path when available. Store that in
`Citation.page_or_section`, for example:

```text
section: Week 2 - RAG & Context Engineering
```

### Jupyter Documents

The current notebook loader flattens markdown and code cells into text. For this slice, keep loader
behavior unchanged unless review decides otherwise. Section-aware chunking can still recognize:

- markdown headings inside flattened notebook text
- fenced code blocks emitted by the notebook loader
- paragraph boundaries

Cell-aware notebook chunking is a possible future improvement, but not required for this slice.

### PDF / Uploaded Files

Keep PDF and upload behavior on `FixedSizeChunker` for this slice unless explicitly enabled later.
The immediate eval corpus is GitHub markdown/notebook content, and over-generalizing across file
formats would expand scope.

---

## 5. Citation And Provenance Requirements

Every chunk must preserve the provenance chain used by the eval scorer:

- `doc_id`
- `title`
- `repo`
- `file_path`
- `commit_hash`
- `line_start`
- `line_end`
- `char_start`
- `char_end`
- `page_or_section` when a section heading is available

The scorer currently matches GitHub gold spans by repo, file path, commit hash, and overlapping line
span. Section-aware chunks must keep those line spans exact enough that the scorer remains valid.

Do not reconstruct citations after retrieval. The chunker owns span capture at ingest time.

---

## 6. Eval Protocol

The existing `eval` collection is the fixed-size baseline. Do not overwrite it.

Create a separate section-aware collection, tentatively:

```text
eval_section
```

Recommended commands after implementation:

```bash
GENACADEMY_CHUNKER=fixed uv run python scripts/eval_retrieval.py \
  --collection eval \
  --json-out eval/runs/phase2-section-baseline.json

GENACADEMY_CHUNKER=section uv run python scripts/ingest_eval_corpus.py \
  --collection eval_section \
  --chunker section \
  --reset-collection

GENACADEMY_CHUNKER=section uv run python scripts/eval_retrieval.py \
  --collection eval_section \
  --json-out eval/runs/phase2-section-aware.json
```

If `--collection` and `--chunker` flags are added, they should default to today's behavior:

- collection: `eval`
- chunker: `fixed`

The section-aware ingest must use the same allowlisted `EVAL_CORPUS` raw sources and commit SHAs as
the baseline. The only intended variable is chunking.

### SQLite Caution

`chunks_meta` uses `chunk_id` as a primary key, and chunk IDs currently derive from
`f"{doc_id}::{ordinal}"`. A section-aware reingest into the same SQLite DB can overwrite fixed-size
chunk metadata for the same docs, even when Chroma uses a different collection.

Implementation should avoid that by one of these approaches:

1. eval alternate ingest writes to a temporary/alternate SQLite path, or
2. eval alternate ingest skips relational metadata if the retrieval eval only needs Chroma metadata, or
3. section-aware chunk IDs include an eval/chunker namespace.

Option 1 is the least invasive for this slice.

---

## 7. Metrics To Report

Commit an eval delta markdown file:

```text
eval/phase2-section-aware-chunking-delta.md
```

Include:

- baseline vs section-aware `recall@k`, `precision@k`, and MRR
- chunk count per collection
- mean/p50/p95 retrieval latency
- per-question movement table
- failure categories helped/hurt/unchanged
- whether q5, q7, q8, q9, and q10 moved
- small-N caveat

Acceptance posture:

- If section-aware improves chunking-stress failures without hurting broad recall, recommend it.
- If it improves recall but lowers precision because chunks get too large/broad, keep it disabled and
  document the tradeoff.
- If it changes little, keep the design notes and drop the implementation from the demo path.

---

## 8. Configuration

Add a minimal chunker toggle:

```text
GENACADEMY_CHUNKER=fixed
GENACADEMY_SECTION_CHUNK_MAX_CHARS=1500
GENACADEMY_SECTION_CHUNK_OVERLAP=150
```

Notes:

- Default remains `fixed` to preserve Phase 0/1 behavior.
- Web/default app should continue to use fixed chunking unless the flag is explicitly set.
- Eval scripts should record the chunker config in JSON output.

---

## 9. Test Strategy

Tests should be offline and source-text based.

Core chunker tests:

- heading path is captured in `page_or_section`
- a markdown table stays with its nearest heading when under `max_chars`
- oversized sections fall back to bounded windows
- fenced code blocks are not split unless too large
- line and char spans are monotonic and overlap the source text exactly
- short documents still produce one full-span chunk
- fixed chunker behavior remains unchanged

Script/eval tests:

- `ingest_eval_corpus.py` accepts collection/chunker arguments and defaults to existing behavior
- `eval_retrieval.py` accepts collection and records chunker config
- alternate section-aware eval does not mutate the baseline `eval` collection
- alternate section-aware ingest does not overwrite baseline SQLite `chunks_meta`

Regression focus:

- q5/q7-style markdown table fixtures should prove the section header and table row appear in the same
  chunk.

---

## 10. Open Review Questions

1. Should section-aware chunking be markdown-only for the first slice, with PDFs/uploads left on
   `FixedSizeChunker`?
2. Should alternate eval ingests use a separate SQLite path or skip SQLite metadata writes entirely?
3. Should `page_or_section` store only the nearest heading or the full heading path?
4. Should the primary delta run disable rerank to isolate chunking, then optionally report a
   chunking+rerank combined run?
5. What maximum section chunk size best balances context completeness against embedding truncation?

---

## 11. Recommended Next Action

Send this design for independent review before writing an implementation plan. The highest-risk point
is not splitting markdown; it is keeping the eval comparison honest while reingesting an alternate
chunked corpus without mutating the fixed-size baseline artifacts.
