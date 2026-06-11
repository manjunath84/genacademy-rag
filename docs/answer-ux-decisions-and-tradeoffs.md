# Answer Trust & Feedback UX — Decisions and Tradeoffs

Companion to `superpowers/specs/2026-06-10-answer-trust-feedback-ux-design.md`
(same series as the phase 0/1/2 decision docs).

## Confidence is a bucket, not a percentage

The grader's 1-5 confidence is self-reported by an LLM (or derived from a cosine
threshold on the fallback path). Showing "80%" would imply calibration that does not
exist. Low/Med/High plus a tooltip stating the basis is the honest rendering.

## Citations merge at presentation time, not in the pipeline

The eval scores per-chunk citations; merging overlapping line ranges in `core/` data
would change the graded contract. So `QueryResult.citations` stays raw and a separate
`sources` field carries the merged presentation rows: two consumers, two shapes, one
source of truth (`retrieved`).

## Snippets come from `retrieved`, not a second lookup

The graph state already carries every retrieved chunk's text. Re-fetching chunk text
from the datastore at render time would add a query per citation and a second
consistency domain. Zero new plumbing: `merge_citations` reads what the graph returns.

## Feedback is an upsert keyed on query and user

Re-clicking flips a verdict instead of stuffing the table. Feedback writes are
best-effort: a DB failure logs an error and the user still gets their answer,
same posture as `log_query`. The posted query id must still refer to a usage row
owned by the current user; forged or stale query ids are rejected before the
best-effort storage path.

## HTMX fragment swap instead of PRG for thumbs

Answers are not addressable URLs (stateless form-post), so a 303 redirect after
feedback would land on `/` and erase the rendered answer. The thumbs POST via HTMX
and swap in a "thanks" fragment; the answer stays on screen.

## Uploaded sources link to a served file, not nothing

Phase 1 already persists upload bytes for re-indexing (`stored_path`), so
`/documents/{doc_id}/file` makes uploaded PDF/PPTX/DOCX sources one-click verifiable
too: the same trust property as the pinned-commit GitHub links. Login-gated; the
path always comes from the datastore row, never the request.

## Overview answers raise the faithfulness stakes

A longer answer has more room to hallucinate. The grounding instructions are
unchanged and the faithfulness eval is re-run with the before/after delta recorded:
"measured, not asserted" applies to prompt changes too.
