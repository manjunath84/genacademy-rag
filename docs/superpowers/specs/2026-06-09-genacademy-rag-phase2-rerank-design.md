# GenAcademy RAG - Phase 2 Cross-Encoder Rerank Design

**Date:** 2026-06-09
**Status:** Phase A design reviewed; implementation completed in PR #3
**Builds on:** Phase 0 + Phase 1 merged on `main` (branch:
`feat/genacademy-rag-phase2-rerank`)
**Source of scope:** `specs/roadmap.md` Phase 2, `docs/design.md` section 6, current
`HybridRetriever`, current eval scripts and `eval/REPORT.md`
**Companion:** `docs/phase2-decisions-and-tradeoffs.md`

---

## 1. Goal

Add one independently droppable Phase 2 depth slice: **cross-encoder reranking** behind the existing
`Retriever` seam, measured as a deterministic before/after retrieval-eval delta.

The success artifact is not "rerank exists." The success artifact is an honest A/B report over the
same immutable eval corpus and same gold set:

- baseline hybrid retrieval: current `recall@k=0.67`, `precision@k=0.22`, `MRR=0.55`
- hybrid + rerank: same metrics, same `top_k=5`, plus retrieval latency cost
- short interpretation of where rerank helped and where it did not

This slice must not touch section-aware chunking, Pinecone, Nebius embeddings, deploy, or any web
surface unless wiring needs an existing setting. It must be removable without destabilizing Phase 0/1.

---

## 2. Sources Read

Local project sources:

- `specs/roadmap.md`: Phase 2 says cross-encoder rerank and section-aware chunking each need a
  before/after eval delta; each item independently droppable.
- `docs/design.md` section 6: `Retriever` seam is `HybridRetriever` now and "+ cross-encoder rerank"
  in Phase 2.
- `src/genacademy_rag/core/retriever.py`: current flow is dense + BM25 candidates,
  RRF fusion, `candidate_k=20`, final `top_k=5`, with a corpus lock and immutable `_Index`.
- `src/genacademy_rag/core/grader.py` and `core/types.py`: `RetrievedChunk.score` is the cosine
  similarity consumed by the grader fallback. It is not a rank score.
- `scripts/eval_retrieval.py`, `scripts/run_eval.py`, `src/genacademy_rag/eval/`, and
  `eval/REPORT.md`: current deterministic eval path and baseline.
- `docs/phase0-decisions-and-tradeoffs.md`, `docs/phase1-decisions-and-tradeoffs.md`: writing style
  and prior invariants.

External docs checked for current library/model behavior:

- Context7 Sentence Transformers docs: `CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")`
  scores `(query, passage)` pairs with `predict(...)`; MS MARCO scores are relevance logits and may
  be sigmoid-normalized only if a 0-1 score is needed.
- Hugging Face Hub model metadata: `cross-encoder/ms-marco-MiniLM-L6-v2` is a
  `sentence-transformers` text-ranking model, Apache-2.0, 22.7M parameters, English, trained on
  `sentence-transformers/msmarco`.

---

## 3. Model Choice

**Decision:** use `cross-encoder/ms-marco-MiniLM-L6-v2` as the default reranker model.

Why this model:

- It is the documented Sentence Transformers MS MARCO reranking example.
- It is small enough for a local CPU path (22.7M parameters) and fits the Phase 2 latency budget.
- It is trained for text-ranking, which matches the job: reorder a short list of query-passage
  candidates, not embed documents.
- It uses the already pinned `sentence-transformers==5.5.1` dependency. No new runtime dependency is
  needed.
- It is Apache-2.0, so there is no license surprise for a portfolio/demo app.

Correct model ID:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The prompt's example had an extra hyphen in `L-6`; implementation should use the documented
`L6-v2` ID.

Score semantics:

- CrossEncoder scores are **reranking logits**.
- They are used only for ordering the candidate pool.
- They are not normalized against cosine or BM25.
- They are not persisted into `RetrievedChunk.score`.
- No grader threshold uses the reranker score.

Local/offline behavior:

- Runtime should support `local_files_only=True` for deterministic/offline operation once the model
  is provisioned.
- Unit tests must never instantiate this model. They use a fake reranker behind a Protocol seam.
- A missing local model should fail with a clear setup error when rerank is enabled, not silently
  fall back to live downloads in eval runs.

---

## 4. Where Rerank Slots

Current `HybridRetriever.retrieve()` shape:

```text
query -> embed(query)
      -> dense_hits = Chroma query top candidate_k
      -> sparse_ids = BM25 top candidate_k
      -> fused = RRF(dense_ids, sparse_ids)
      -> ranked = top_k by RRF
      -> RetrievedChunk(score = cosine similarity)
```

Rerank should slot **after RRF fusion and before final `top_k` truncation**, over the **full fused
candidate union** (revised per independent review):

```text
query -> embed(query)
      -> dense_hits = Chroma query top candidate_k
      -> sparse_ids = BM25 top candidate_k
      -> fused = RRF(dense_ids, sparse_ids)
      -> pool = ALL fused candidates (union of both lists, <= 2 * candidate_k),
                optionally capped by rerank_pool when set
      -> rerank(query, pool chunks)
      -> ranked = top_k by CrossEncoder score (stable sort; ties keep RRF order)
      -> RetrievedChunk(score = original cosine similarity)
```

Details:

- Dense and BM25 still generate the recall-oriented pool; each list is already capped at
  `candidate_k`, so the fused union is bounded at `2 * candidate_k` (40 pairs worst case).
- The cross-encoder reranks that full fused union, not the full corpus.
- `rerank_pool` is a setting; default (unset/0) means the full fused union. A smaller value
  truncates the pool by RRF rank, for latency control on larger corpora only.
- If rerank is disabled, behavior remains today's exact RRF -> `top_k` flow.
- If a candidate is BM25-only, `RetrievedChunk.score` remains `0.0`, same as today.

Why not rerank dense results only:

- It would throw away the Phase 0 exact-match/BM25 value before rerank sees it.
- The existing eval includes exact-match questions specifically to test sparse retrieval.

Why not rerank after `top_k=5`:

- Too late. The candidate that should move into the answer context may already have been cut.

Why not truncate to top `candidate_k` by RRF before rerank (the original draft's choice):

- The eval corpus is 53 chunks; the fused union is at most 40 candidates. Truncating to
  top-20-by-RRF means a candidate RRF-ranked 21+ can never be rescued by the cross-encoder —
  which defeats rerank's purpose, since RRF mis-ranking is exactly what rerank exists to fix.
- The marginal cost is ~20 extra pairs at ~0.8 ms/pair (~16 ms), trivially inside the latency
  budget. On a corpus where the union grows, `rerank_pool` restores the bound explicitly.

---

## 5. Core Shape

Add a pure-core reranker seam, likely in `src/genacademy_rag/core/reranker.py`:

```python
class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]: ...
```

Implementation:

- `SentenceTransformersCrossEncoderReranker` imports `CrossEncoder` inside `__init__`.
- It builds pairs as `[(query, chunk.text), ...]`.
- It calls `model.predict(pairs, batch_size=settings.rerank_batch_size)`.
- It returns chunks ordered by descending score, with scores kept internal to the retriever path.
- **Tie-break determinism requirement:** the rerank sort must be a stable sort over the
  deterministic RRF-ordered input, so equal rerank scores preserve RRF order. The retrieval
  pipeline must produce an identical ranking on repeated runs over the same corpus and query.

Testing seam:

- `FakeReranker` returns deterministic scores from a supplied map or callable.
- Tests can prove rerank happens after RRF by constructing a case where RRF order and fake rerank
  order disagree.
- Tests can prove `RetrievedChunk.score` is still cosine by making the fake reranker score very high
  for a chunk whose dense cosine is known.

No FastAPI/template imports are allowed in `core/` or `data/`.

---

## 6. Configuration and Toggle

Rerank is disabled by default. Baseline-vs-rerank should be one setting flip:

```text
GENACADEMY_RERANK_ENABLED=false
GENACADEMY_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
GENACADEMY_RERANK_LOCAL_FILES_ONLY=true
GENACADEMY_RERANK_BATCH_SIZE=32
GENACADEMY_RERANK_POOL=0
GENACADEMY_RERANK_DEVICE=
GENACADEMY_RERANK_CACHE_DIR=
```

Notes:

- `GENACADEMY_RERANK_ENABLED=false` preserves Phase 0/1 behavior by default.
- `GENACADEMY_RERANK_POOL=0` (default) reranks the full RRF-fused candidate union
  (`<= 2 * candidate_k`). A positive value truncates the pool by RRF rank — a latency knob for
  larger corpora, not the default.
- `GENACADEMY_RERANK_LOCAL_FILES_ONLY=true` protects reproducible eval runs from accidental network
  downloads. With this set, a fresh clone without the cached model must fail with a clear setup
  error pointing at the provisioning task — never silently download during an eval run.
- **Model provisioning is an explicit, one-time task**, not a side effect: a documented command
  (e.g. `uv run python scripts/provision_rerank_model.py`, or an equivalent documented
  `huggingface-cli download` step) that fetches the model into the cache dir. It is the only place
  `local_files_only=false` is acceptable.
- `GENACADEMY_RERANK_DEVICE` is optional and stays configurable at runtime; when empty, Sentence
  Transformers chooses (`mps` on this machine). **Committed eval-delta runs pin
  `GENACADEMY_RERANK_DEVICE=cpu`** so the committed artifact is reproducible across runs and
  machines — MPS/CUDA float math can perturb near-tie orderings.
- No secrets are introduced. Existing generation API keys remain env-only.

The web app and scripts should construct the reranker only when enabled and pass it into
`HybridRetriever`. The retriever itself should not read environment variables.

---

## 7. Score-Semantics Invariant

This is the load-bearing invariant:

```text
RetrievedChunk.score == cosine similarity from VectorStore.query
```

It remains true with rerank enabled.

Rationale:

- `core/grader.py` fallback computes `top = max(r.score for r in retrieved)` and compares it to the
  cosine threshold.
- RRF scores are tiny rank-fusion numbers and would make the fallback refuse nearly everything.
- CrossEncoder logits are a different scale again and are not calibrated for the existing threshold.

Implementation rule:

- Keep reranker scores in local variables or an internal dataclass such as
  `_RerankedCandidate(chunk_id, rerank_score)`.
- When returning `RetrievedChunk`, use `sim_by_id.get(cid, 0.0)` exactly as the current retriever
  does.
- Do not add a `score` overload or mutate `RetrievedChunk`.

One behavioral consequence the invariant does **not** prevent (state it, don't hide it):

- The grader fallback computes `max(r.score for r in retrieved)` over whichever chunks occupy the
  final `top_k`. Rerank changes that membership. If the reranker promotes BM25-only chunks
  (which carry `0.0`) into top_k and displaces dense hits, the max cosine seen by the fallback
  drops, and a question that answered under baseline can refuse under rerank (or vice versa).
- This is accepted: the invariant protects the *field's semantics*, not fallback-path behavioral
  identity. The eval-delta report must note any refusal-behavior changes attributable to top_k
  membership shifts, and a regression test must pin the mechanism (fake reranker promotes a
  BM25-only chunk; assert the fallback grade follows the new top_k's max cosine).

Regression tests must cover:

- reranker score does not appear in `RetrievedChunk.score`
- BM25-only candidate still carries `0.0`
- grader cosine fallback still answers/refuses based on cosine, not reranker logits
- rerank-induced top_k membership change flows through to the fallback's max-cosine as designed

---

## 8. Latency Budget

The project latency ceiling in `docs/design.md` is under 8 seconds end-to-end. Generation dominates
that budget; rerank must remain a small local CPU add-on.

Measured on this machine during Phase A:

- Current baseline eval run: `uv run python scripts/eval_retrieval.py` completed and reproduced
  `recall@k=0.67`, `precision@k=0.22`, `MRR=0.55`.
- Local-only cached model load for `cross-encoder/ms-marco-MiniLM-L6-v2`: about `590 ms`.
- Warm CrossEncoder scoring over 20 query-passage pairs: about `16 ms` (`0.8 ms/pair`).
- A prior run that included one-time download/provisioning took about `17.6 s`; that is not part of
  steady-state request latency.

Budget:

- Startup/model-load cost is acceptable if the model loads once at process startup or first enabled
  retrieval.
- Per-query rerank cost target: under `150 ms` p95 over the eval set for the full fused union
  (`<= 2 * candidate_k = 40` pairs; ~32 ms extrapolated from the 0.8 ms/pair measurement) on local
  CPU hardware.
- If measured p95 exceeds `300 ms`, keep rerank disabled by default and document the cause in the
  delta report.

The Phase C eval report must include actual measured retrieval latency on real eval questions, not
only the synthetic 20-pair timing above.

---

## 9. Exact Eval-Delta Protocol

The eval corpus and gold set stay fixed:

- Chroma collection: `eval`
- Gold set: `src/genacademy_rag/eval/gold/gold_set.yaml`
- Metrics: recall@k, precision@k, MRR from `src/genacademy_rag/eval/retrieval_eval.py`
- `top_k=5`
- `candidate_k=20`
- no writes to the `eval` collection

Baseline command:

```bash
GENACADEMY_RERANK_ENABLED=false uv run python scripts/eval_retrieval.py \
  --json-out eval/runs/phase2-rerank-baseline.json
```

Rerank command (committed runs pin `device=cpu` for reproducibility; the device stays
configurable for non-committed local use):

```bash
GENACADEMY_RERANK_ENABLED=true GENACADEMY_RERANK_DEVICE=cpu \
  uv run python scripts/eval_retrieval.py \
  --json-out eval/runs/phase2-rerank-enabled.json
```

Expected script additions in Phase C:

- `--json-out PATH` writes aggregate metrics, per-question rows, and latency fields.
- The default printed one-line summary remains for continuity with Phase 0.
- Latency fields include at least `retrieval_ms_mean`, `retrieval_ms_p50`, `retrieval_ms_p95`, and
  per-question `retrieval_ms`.
- Rerank run records `rerank_enabled=true`, `rerank_model`, `candidate_k`, `top_k`,
  `rerank_pool` (0 = full union), `rerank_device`, `rerank_batch_size`, and
  `rerank_local_files_only`.

Committed delta artifact:

```text
eval/phase2-rerank-delta.md
```

That markdown file should contain:

- environment/config snapshot relevant to retrieval (including the pinned `device=cpu`)
- baseline vs rerank metrics table
- latency table
- per-question movement table showing recall/MRR changes and top retrieved source movement
- short interpretation: helped, hurt, unchanged, and why
- **small-N caveat stated explicitly:** only ~12 retrieval-scored questions exist, so a single
  question flipping moves aggregate recall/MRR by roughly 0.08. Per-question movement is the
  meaningful evidence; aggregate deltas smaller than one question's worth are noise, and no
  statistical-significance claim is honest at this N.

Acceptance posture:

- If rerank improves MRR/precision without reducing recall, it can be recommended for demo use.
- If rerank improves ranking but hurts recall, leave it disabled by default and report the tradeoff.
- If rerank does not improve the eval, still keep the implementation behind the flag only if tests
  prove it is inert when disabled and the report says not to enable it. Otherwise drop the slice.

---

## 10. Test Strategy for the Future Plan

Tests should be TDD and offline:

- `tests/core/test_reranker.py`
  - fake CrossEncoder/model object scores `(query, text)` pairs deterministically
  - wrapper sorts descending by score
  - empty input returns empty output
  - no live Hugging Face calls
- `tests/core/test_retriever.py`
  - disabled rerank preserves existing RRF behavior
  - enabled rerank reorders the RRF candidate pool before `top_k`
  - rerank sees the full fused union (a candidate RRF-ranked below `top_k`, present in only one
    source list, can be rescued into the final `top_k`)
  - `rerank_pool=N` truncates the pool by RRF rank when set
  - tie-break determinism: equal rerank scores preserve RRF order (stable sort); repeated
    retrieval over the same corpus and query yields an identical ranking
  - `RetrievedChunk.score` remains cosine similarity, not reranker score
  - BM25-only hits still return score `0.0`
- `tests/core/test_grader.py`
  - cosine fallback still uses `RetrievedChunk.score` correctly with reranked retrieval
  - rerank-induced top_k membership change (fake reranker promotes a BM25-only chunk) flows
    through to the fallback's max-cosine as designed (section 7)
- `tests/test_config.py`
  - new rerank env settings parse correctly
  - default disabled
- `tests/eval/` or script-level tests
  - JSON output includes metrics and latency
  - eval script uses `eval` collection and does not mutate it

Verification commands for Phase C:

```bash
uv run pytest
uv run ruff check src tests scripts
GENACADEMY_RERANK_ENABLED=false uv run python scripts/eval_retrieval.py --json-out eval/runs/phase2-rerank-baseline.json
GENACADEMY_RERANK_ENABLED=true GENACADEMY_RERANK_DEVICE=cpu uv run python scripts/eval_retrieval.py --json-out eval/runs/phase2-rerank-enabled.json
```

---

## 11. Open Review Questions

1. ~~Should rerank use the top `candidate_k` after RRF, or the full dense/sparse union capped by
   `2 * candidate_k`?~~ **Resolved by independent review: rerank the full fused union** (default
   `rerank_pool=0`), because truncating a 53-chunk corpus's <= 40-candidate union to top-20-by-RRF
   makes RRF-rank-21+ candidates unrescuable at a trivial ~16 ms marginal cost. See section 4.
2. Should the model be loaded lazily on the first reranked query or eagerly during app/eval startup?
   This design prefers lazy construction at wiring time when the flag is enabled, then reuse.
3. Should the eval script fail hard when `GENACADEMY_RERANK_ENABLED=true` and the model is not cached
   with `local_files_only=true`? This design says yes, because silent network downloads break
   reproducibility.
4. Is the acceptance posture acceptable if rerank improves MRR but does not improve recall? This
   design treats MRR/precision improvement with stable recall as a valid rerank win because rerank
   mainly reorders, not expands, the pool.
