# GenAcademy RAG - Learnings So Far

**Date:** 2026-06-09
**Scope:** lessons from Phase 0, Phase 1, and the first Phase 2 slice through PR #3.
**Primary sources:** `docs/design.md`, `specs/roadmap.md`,
`docs/phase0-decisions-and-tradeoffs.md`, `docs/phase1-decisions-and-tradeoffs.md`,
`docs/phase2-decisions-and-tradeoffs.md`, `eval/REPORT.md`, and
`eval/phase2-rerank-delta.md`.

## Executive Summary

The biggest project learning is that a RAG app is only as credible as its eval and its refusal path.
The model is not the center of gravity. Chunking, retrieval quality, citation provenance, and honest
failure handling are the parts that determine whether the system is trustworthy.

This project started as a week-sized Gen Academy knowledge assistant, but the durable engineering
lesson is broader: keep the graded spine small, make every quality claim measurable, and put scope
behind seams so improvements can be swapped in one at a time.

## Product And Scope

1. **A complete boring spine beats an impressive partial demo.**
   The useful product loop is: ingest course materials, retrieve grounded chunks, answer with
   citations, or refuse. Admin features, reranking, dashboards, and deploy work only matter after that
   loop is real and evaluated.

2. **The corpus needs two tiers.**
   The eval corpus must be frozen and commit-pinned, while the production corpus can grow through
   uploads and repo updates. Mixing these would make scores drift every time content changes.

3. **The sample-solution firewall needs to fail closed.**
   The Week 2 reference solution must never inform this build. An allowlist is safer than a denylist:
   if a repo is not explicitly approved, it cannot be fetched.

4. **Phase boundaries are a risk-control tool.**
   Phase 0 protected the graded assistant and eval. Phase 1 added product operations. Phase 2 adds
   depth only as independently droppable slices. This kept nice-to-have work from weakening the
   deliverable.

## RAG And Retrieval

1. **Hybrid retrieval earned its complexity.**
   Dense retrieval is good for semantic matching, but it misses exact tokens and rare terms. BM25
   covers the lexical path. Reciprocal Rank Fusion avoided score-normalization problems between cosine
   similarity and BM25.

2. **Score fields are contracts.**
   `RetrievedChunk.score` means dense cosine similarity because the grader fallback uses it as an
   answerability signal. Phase 2 reranking improved order, but reranker logits stayed internal because
   they are not calibrated cosine scores.

3. **Rerank belongs after recall, not instead of recall.**
   The cross-encoder should reorder a bounded candidate union produced by dense + BM25 + RRF. It should
   not replace the recall stage, rerank only the final top 5, or scan the whole corpus.

4. **Rerank quality and latency both matter.**
   PR #3 improved retrieval from recall@k 0.67 to 0.79, precision@k 0.22 to 0.25, and MRR 0.55 to
   0.58. The cost was material: CPU-pinned p95 retrieval latency rose from about 286 ms to 886 ms in
   the committed run. That makes default-off the right posture.

5. **Chunking is still the next quality lever.**
   The current eval failures repeatedly point at table rows, section headers, and answers split across
   fixed-size chunk boundaries. Section-aware chunking is likely a better next Phase 2 slice than more
   retrieval tuning.

## Evaluation

1. **The deterministic retrieval eval is the protected artifact.**
   Recall@k, precision@k, and MRR over a fixed gold set are reproducible and directly grade retrieval
   quality. The LLM judge is useful, but secondary.

2. **Small evals require humility.**
   The retrieval-scored set has n=12, so one question moves aggregate recall or MRR by about 0.08.
   Per-question movement is more informative than treating aggregate deltas as statistically strong.

3. **Unanswerable questions are load-bearing.**
   A RAG assistant that only answers answerable questions has not tested the product promise. The
   refusal path needs its own eval coverage because hallucination often appears as over-answering.

4. **Failure taxonomy makes the report actionable.**
   Labels like `ChunkingBoundary`, `TopKTooSmall`, `RefusalFalsePositive`, and
   `RetrievalRecallFailure` turn scores into engineering direction. They pointed directly toward
   overlap, section-aware chunking, and reranking experiments.

5. **The faithfulness number depends on scorer choice.**
   LLM-as-judge can catch incomplete or unsupported answers, but it costs live calls and can fail under
   throttling. Citation-grounding is less rich but deterministic. Reports must state which scorer was
   used.

## Architecture

1. **Pure core / thin view paid off.**
   Keeping FastAPI, templates, and sessions out of core RAG logic made retrieval, grading, eval, and
   datastore behavior testable without a live web app.

2. **Pluggability should be interface + config, not conditionals.**
   Provider, vector store, retriever, chunker, and datastore seams let the project add Phase 2 depth
   without scattering `if provider == ...` logic through business code.

3. **Use an abstraction only when the change axis is real.**
   `Chunker`, `VectorStore`, `Retriever`, and `ModelProvider` were justified because the roadmap
   names specific swaps. Splitting `Datastore` early was deferred because no second backend existed yet.

4. **Cross-store consistency is a real design problem even in a small app.**
   Upload/delete touches Chroma, SQLite, filesystem, and the retriever's BM25 snapshot. The safe design
   is serialized corpus mutation with an ordering that fails toward "not searchable" rather than
   serving deleted or untracked content.

5. **Runtime defaults should preserve the green baseline.**
   Rerank is disabled by default. Local/offline files are preferred for deterministic eval. Expensive
   or network-dependent features should require explicit opt-in.

## Security And Auth

1. **Bearer credentials need password-grade treatment.**
   Invite codes are bearer credentials. They are shown once and stored as bcrypt hashes, not plaintext.

2. **Salted hashes are not lookup keys.**
   Bcrypt produces a different hash for the same secret each time. The fix was an invite format with a
   clear lookup id plus a secret half that is bcrypt-verified.

3. **RBAC should be enforced at every admin boundary.**
   Admin pages, upload routes, delete routes, invite generation, and revoke flows all need role checks,
   not just hidden navigation.

4. **CSRF is required for destructive form posts.**
   Session auth plus server-rendered forms still need CSRF tokens on admin mutations.

5. **Default secrets should warn loudly.**
   The development session secret is acceptable locally but should produce warnings so it is not
   accidentally deployed.

## Error Handling And Refusal

1. **Parsing security-critical booleans must fail safe.**
   JSON-mode models can emit `"false"` as a string. In Python, `bool("false")` is `True`, which can
   turn a refusal into an answer. Strict parsing plus fallback is the right pattern.

2. **Fallbacks must be honest and observable.**
   The grader can fall back to a cosine threshold, but the result should carry whether fallback was
   used. Silent fallback makes debugging eval failures harder.

3. **A refusal path is product behavior, not a UI message.**
   The system must decide answerability before generation and prevent unsupported answers, rather than
   relying on the answer model to be modest.

4. **Rerank can change refusal behavior indirectly.**
   Even when score semantics are preserved, rerank changes which chunks are in top-k. If it promotes
   BM25-only chunks with score 0.0 and displaces dense hits, the cosine fallback can change. That is an
   expected mechanism that needs tests and reporting.

## Testing And Review

1. **Every non-trivial invariant needs a regression test.**
   The strongest tests covered eval isolation, score semantics, invite redemption, delete consistency,
   CSRF, admin access, rerank ordering, and fallback behavior.

2. **Reviewer separation caught real issues.**
   Fresh review found the boolean parsing bug, invite-code lookup flaw, corpus mutation consistency
   problem, and stale documentation gate language. Builder self-review would likely have missed some of
   these.

3. **Tests should use fakes for local determinism.**
   Reranker unit tests use a fake CrossEncoder and never instantiate the real Hugging Face model. The
   live model is reserved for explicit eval runs.

4. **Evidence before done changes behavior.**
   The project standard of showing ruff, pytest, eval output, and review findings prevented "it should
   work" from becoming the definition of done.

5. **Docs are review surface.**
   Stale status text can contradict the process even if the code is correct. Reviewers should treat
   docs as part of the system, not an afterthought.

## Agent Workflow

1. **Plan gates reduced thrash.**
   The strongest work happened when the sequence was design -> independent review -> implementation
   plan -> tests -> code -> verification -> review. Jumping straight to code would have hidden design
   questions inside implementation.

2. **Second-model critique is useful when it has concrete artifacts.**
   Reviews were most valuable when they inspected diffs, line references, eval outputs, and failure
   modes rather than giving generic advice.

3. **Keep agent changes surgical.**
   Narrow PRs made review possible: Phase 0 spine, Phase 1 product layer, then one Phase 2 rerank
   slice. Broad refactors would have made eval regressions harder to attribute.

4. **Prompts should force measurable outputs.**
   Useful agent prompts asked for gates, tradeoffs, tests, exact source references, and eval deltas.
   Vague prompts produce plausible prose; measurable prompts produce artifacts.

5. **Use docs as memory between contexts.**
   Decision and tradeoff documents made later sessions faster because they captured not only what was
   built, but why alternatives were rejected.

## What Changed Our Mind

1. **Dense-only was too weak for exact-match questions.**
   The eval design made BM25 non-optional.

2. **Lock-free retriever snapshots were not enough.**
   Dense retrieval reads live Chroma while sparse retrieval reads an in-memory BM25 snapshot. Because
   those are two consistency domains, a simple atomic snapshot swap did not fully protect deletes.

3. **Invite-code hashing needed an id/secret shape.**
   "Store bcrypt hash and look it up" looked secure but was not lookupable. The API-token-shaped fix
   was both secure and practical.

4. **Reranking the top-k was too late.**
   A cross-encoder can only rescue candidates it sees. The pool had to be the full fused union before
   top-k truncation.

5. **Latency claims need real eval text, not synthetic snippets.**
   Synthetic pair scoring looked cheap, but full 1000-character chunks on CPU made p95 much higher.

## Next Best Learning Target

Section-aware chunking is the highest-signal next experiment. The current failure table repeatedly
points at fixed-size chunk boundaries, table context loss, and section headers separated from their
content. The right next artifact is a design and implementation plan for `SectionAwareChunker`, with
the same before/after eval delta discipline used for reranking.
