# Phase 2 - Cross-Encoder Rerank Decisions & Tradeoffs

**Purpose:** an interview-prep artifact for the first Phase 2 slice. Every non-trivial rerank fork is
written as **Problem -> Options -> Tradeoffs -> Decision -> How an interviewer probes it**. The
implementation spec is `docs/superpowers/specs/2026-06-09-genacademy-rag-phase2-rerank-design.md`.

**Status:** design written, pending independent review. No implementation plan or code should happen
until the review gate approves the design.

**Scope:** cross-encoder reranking only. Section-aware chunking, Pinecone, Nebius embeddings, deploy,
and web/product changes are separate Phase 2 items.

---

## Decision 1 - Add rerank now, but behind a flag and measured as an eval delta

**Problem.** Phase 0 already has hybrid dense + BM25 retrieval. Why add a cross-encoder at all?

**Options.** (a) keep hybrid only; (b) replace hybrid with a cross-encoder; (c) keep hybrid as the
candidate generator and add cross-encoder rerank behind a flag.

**Tradeoffs.** Hybrid retrieval is cheap and already gives recall from two complementary signals:
dense semantic search plus BM25 lexical search. But it fuses ranks with RRF, which is deliberately
coarse. A cross-encoder can inspect the query and full candidate text jointly, which is better for
ordering the final answer context. Replacing hybrid would be wrong because a cross-encoder cannot scan
the full corpus efficiently; it needs a small candidate set first. Adding it unconditionally would risk
latency and could regress a green Phase 0/1 spine.

**Decision: keep hybrid as the recall layer; add cross-encoder rerank as an optional Phase 2 ordering
layer.** It is off by default and measured with the same deterministic eval. The feature is valuable
only if the before/after table shows a real ranking benefit, especially MRR/precision with recall held
steady.

**How an interviewer probes it.**

- *"Why not just tune RRF?"* -> RRF has no semantic understanding of the query-passage pair; it only
  combines rank positions. A cross-encoder jointly reads the query and passage, so it can make a
  relevance judgment on the final candidate pool.
- *"What if it makes metrics worse?"* -> the flag stays off and the eval report says so. Phase 2 items
  are independently droppable; preserving the green Phase 0/1 baseline matters more than forcing a
  demo feature.
- *"Why is this an eval-delta slice?"* -> rerank is a retrieval-quality claim. If recall/precision/MRR
  do not move, the claim is not defensible.

---

## Decision 2 - Cross-encoder vs bi-encoder

**Problem.** What model shape should score relevance after candidates are found?

**Options.** (a) another bi-encoder/dense model; (b) a cross-encoder; (c) LLM-as-reranker.

**Tradeoffs.** A bi-encoder embeds query and document separately, so document embeddings can be
precomputed and search stays fast. That is exactly why Phase 0 uses local embeddings. But a bi-encoder
has already had its chance in the dense retrieval stage; adding another one mostly duplicates the
same kind of signal. A cross-encoder processes `(query, passage)` together, which is slower but more
accurate for pairwise ranking. An LLM-as-reranker is flexible but introduces network calls, cost,
nondeterminism, and a second live-model dependency in the protected eval path.

**Decision: local Sentence Transformers cross-encoder.** Use the bi-encoder for candidate generation
and the cross-encoder for final ordering over the fused candidate union (at most ~40 candidates).
No LLM calls enter the retrieval eval.

**How an interviewer probes it.**

- *"Why not use the answer LLM to pick the best chunks?"* -> that makes the deterministic retrieval
  eval depend on a live generation model. The protected metric needs to stay local, cheap, and
  reproducible.
- *"Why is cross-encoder slower?"* -> it cannot precompute document vectors independently. It runs a
  transformer forward pass for every query-passage pair, so cost scales with candidate count.
- *"Why is it more accurate?"* -> joint attention lets the model compare query terms and passage terms
  in context, instead of comparing two separately compressed vectors.

---

## Decision 3 - Use `cross-encoder/ms-marco-MiniLM-L6-v2`

**Problem.** Which local cross-encoder model is the right default for a small portfolio RAG app?

**Options.** (a) `cross-encoder/ms-marco-MiniLM-L6-v2`; (b) a larger MS MARCO cross-encoder such as
an L12 variant; (c) a domain-specific or newly fine-tuned reranker.

**Tradeoffs.** Larger rerankers may improve quality, but this app has a small eval corpus, a strict
local/offline requirement, and an 8-second end-to-end latency ceiling. Fine-tuning is unjustified
without more labeled data. `ms-marco-MiniLM-L6-v2` is the documented Sentence Transformers reranking
example, is small enough for local CPU/MPS use, has an Apache-2.0 license, and uses the existing
pinned `sentence-transformers==5.5.1` dependency.

**Decision: `cross-encoder/ms-marco-MiniLM-L6-v2`.** Keep the model configurable, but use this exact ID
as the default.

**How an interviewer probes it.**

- *"Why MS MARCO for course docs?"* -> the task is general passage retrieval: given a natural-language
  query, rank passages. MS MARCO is a standard training source for that shape. The eval delta, not
  model reputation, decides whether it works for this corpus.
- *"What does the score mean?"* -> it is a relevance logit for ordering. It is not cosine similarity
  and not a calibrated answerability confidence.
- *"Why not fine-tune?"* -> the project has 15 gold questions, not a reranker training set. Fine-tuning
  would overfit and add scope without evidence.

---

## Decision 4 - Rerank after RRF fusion, before `top_k` truncation

**Problem.** Where exactly does rerank belong in the retriever?

**Options.** (a) rerank dense candidates before BM25/RRF; (b) rerank after RRF but before final
`top_k`; (c) rerank only the final top 5; (d) rerank everything in the corpus.

**Tradeoffs.** Reranking dense candidates before BM25 throws away the lexical exact-match path.
Reranking only the final top 5 is too late; a relevant candidate ranked 6-20 can never move into the
answer context. Reranking the full corpus is infeasible in general. The current retriever already has
the correct two-stage shape: dense and sparse recall first, RRF to produce a bounded candidate pool,
then final ordering. An earlier draft truncated the pool to top-`candidate_k`-by-RRF before rerank;
independent review flagged that this makes any candidate RRF-ranked 21+ unrescuable — defeating
rerank's purpose, since RRF mis-ranking is exactly what rerank exists to fix — at a saving of only
~20 pairs (~16 ms at 0.8 ms/pair) on a 53-chunk corpus.

**Decision: rerank the full RRF-fused candidate union (<= 2 * candidate_k pairs) after fusion and
before `top_k=5` truncation; `rerank_pool` is a setting defaulting to the full union.**
Dense/BM25/RRF remain the recall stage; cross-encoder is the precision/order stage.

**How an interviewer probes it.**

- *"Why not rerank after top_k?"* -> because rerank cannot recover candidates that were already cut.
- *"Why not feed it BM25-only candidates too?"* -> we do. BM25 and dense are fused first, then the
  full fused union is reranked.
- *"What bounds latency?"* -> each source list is capped at `candidate_k=20`, so the cross-encoder
  sees at most ~40 pairs per query (~32 ms warm on CPU), not the full collection. On a larger
  corpus, `rerank_pool` truncates by RRF rank explicitly instead of silently.
- *"Why not truncate by RRF first?"* -> truncating before rerank reintroduces the exact failure
  mode rerank is meant to fix: a relevant candidate that RRF under-ranked can never be rescued.

---

## Decision 5 - Preserve `RetrievedChunk.score` as cosine similarity

**Problem.** Once rerank introduces a new score, where does it go?

**Options.** (a) overwrite `RetrievedChunk.score` with reranker score; (b) overwrite it with RRF
score; (c) keep `RetrievedChunk.score` as cosine and carry rerank score only internally.

**Tradeoffs.** Overwriting the score is tempting because the returned chunks are now ordered by
reranker relevance. But `core/grader.py` uses `RetrievedChunk.score` as the fallback answerability
signal: `max(score) >= cosine_threshold`. RRF scores and cross-encoder logits live on unrelated
scales. Feeding either into the fallback would silently change refusal behavior and can make the bot
answer/refuse for the wrong reason.

**Decision: `RetrievedChunk.score` remains cosine similarity from `VectorStore.query`.** Reranker
scores are local ordering data, not the public confidence field.

**How an interviewer probes it.**

- *"Why not expose the better reranker score?"* -> because the existing field has a contract. It is
  consumed by the grader fallback as cosine. If we need rerank observability later, add a separate
  field or debug artifact, not an overload.
- *"What happens if you put logits into the cosine threshold?"* -> the fallback threshold becomes
  meaningless. A high reranker logit is not a cosine similarity; a negative logit could still be one of
  the best candidates.
- *"How do you enforce it?"* -> regression tests set fake reranker scores that differ obviously from
  cosine and assert the returned `RetrievedChunk.score` remains the original cosine value.
- *"Does the invariant make rerank invisible to the refusal path?"* -> no, and that is stated, not
  hidden. The fallback takes `max(score)` over whichever chunks occupy the final top_k; rerank
  changes that membership. If rerank promotes BM25-only chunks (score `0.0`) and displaces dense
  hits, the max cosine drops and fallback refusal behavior can change. The field's semantics are
  protected; behavioral identity of the fallback is not, and the eval delta must report any such
  refusal changes.

---

## Decision 6 - Keep eval immutable and reproducible

**Problem.** How do we prove rerank helped without moving the benchmark under our feet?

**Options.** (a) re-ingest the eval corpus with rerank enabled; (b) run rerank over the existing
`eval` Chroma collection and same gold set; (c) test manually in the web app.

**Tradeoffs.** Re-ingesting changes the input and can hide regressions behind corpus drift. Manual web
testing is useful for demos but not a retrieval metric. The existing deterministic eval was built
specifically to avoid this trap: fixed Chroma collection, fixed commit hashes, fixed YAML gold set,
fixed scoring code.

**Decision: run both baseline and rerank against the existing immutable `eval` collection and
`gold_set.yaml`.** The only intended variable is the rerank flag.

**How an interviewer probes it.**

- *"How do you know the improvement is from rerank?"* -> same corpus, same embeddings, same gold set,
  same `top_k`, same scoring code. One flag changes.
- *"Can uploads inflate the score?"* -> no. The eval script reads the `eval` collection, and the
  scorer requires matching `commit_hash` provenance.
- *"Why include latency in the eval delta?"* -> rerank trades compute for ranking quality. A quality
  delta without a latency delta is an incomplete engineering claim.
- *"Is the delta statistically meaningful?"* -> the honest answer is stated in the report: only ~12
  retrieval-scored questions exist, so one question flipping moves aggregate recall/MRR by ~0.08.
  Per-question movement is the evidence; aggregate deltas smaller than one question's worth are
  noise, and no significance claim is made at this N.
- *"How is the run itself reproducible?"* -> committed eval-delta runs pin
  `GENACADEMY_RERANK_DEVICE=cpu` (MPS/CUDA float math can perturb near-tie orderings), and the
  rerank sort is a stable sort over deterministic RRF order, so ties cannot reorder between runs.

---

## Decision 7 - Local/offline by default, fake reranker in tests

**Problem.** The selected reranker lives on Hugging Face. How do we keep tests and eval deterministic?

**Options.** (a) let Sentence Transformers download as needed; (b) require local cached files for
rerank eval/runtime; (c) mock only the network layer while still constructing the real model in tests.

**Tradeoffs.** Auto-downloads are convenient but make eval runs depend on network availability and Hub
state. Constructing the real model in unit tests is slow and brittle. Requiring local files makes the
runtime failure explicit: if the model is not provisioned, rerank-enabled eval should fail clearly
before reporting metrics.

**Decision: rerank runtime supports `local_files_only=true`; tests use a fake reranker and never load
the real model.** Model provisioning is an explicit one-time task (a documented script or
`huggingface-cli download` command — the only place `local_files_only=false` is acceptable), not a
side effect hidden inside tests or eval runs. On a fresh clone with `local_files_only=true` and no
cached model, rerank-enabled runs fail with a clear setup error naming the provisioning task.

**How an interviewer probes it.**

- *"How do you test rerank offline?"* -> a `Reranker` Protocol plus `FakeReranker` that returns
  deterministic scores. Unit tests verify ordering and score invariants without live calls.
- *"What if the model is not cached?"* -> rerank-enabled eval fails with a clear setup error. Baseline
  eval still runs with rerank disabled.
- *"Why not use HF_TOKEN?"* -> no secret is needed for this public model, and secrets should not be
  part of deterministic retrieval eval.

---

## Decision 8 - Latency budget: measure steady-state rerank, not download time

**Problem.** How expensive is reranking and what cost belongs in the product budget?

**Options.** (a) include one-time model download in query latency; (b) include model load in every
query; (c) measure steady-state forward-pass latency and record model-load separately.

**Tradeoffs.** Download time is a provisioning concern, not a request-path cost. Loading the model per
query would be a design bug. The honest product measurement is: load once, reuse, and measure the
extra forward pass over the actual candidate pool.

**Decision: report both startup/model-load cost and per-query retrieval latency, but judge the request
budget on warm rerank latency.** Phase A measurement on this machine showed about `590 ms` local-only
cached model load and about `16 ms` warm scoring for 20 pairs (`0.8 ms/pair`; the full fused union of
<= 40 pairs extrapolates to ~32 ms); Phase C must measure real eval-question latency on the pinned
`device=cpu` configuration and commit it in `eval/phase2-rerank-delta.md`.

**How an interviewer probes it.**

- *"Does 16 ms prove production latency?"* -> no. It is a local synthetic sanity check. The committed
  eval delta must measure real candidate texts and report p50/p95.
- *"What if the model load is slow?"* -> load once at startup or first enabled query; do not load per
  request. If cold-start matters for deploy, document it separately.
- *"What if p95 is too high?"* -> keep rerank disabled by default, lower candidate count only with a
  new eval, or drop the slice.

---

## Cross-cutting themes

1. **Separate recall from ordering.** Dense + BM25 + RRF builds the candidate set; cross-encoder
   rerank orders it. Mixing those jobs makes the system harder to reason about.
2. **A score field is a contract.** `RetrievedChunk.score` already means cosine. Changing that would
   be a behavioral regression in the refusal path, even if retrieval metrics looked better.
3. **One-variable evals are credible evals.** Same corpus, same gold set, same scorer, one flag.
4. **Default-off is a valid Phase 2 posture.** A measured but disabled feature is preferable to
   weakening a green baseline.
5. **Local deterministic beats impressive live dependencies.** Rerank is a retrieval component, so it
   belongs in the LLM-free deterministic eval spine.
