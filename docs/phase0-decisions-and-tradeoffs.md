# Phase 0 — Decisions & Tradeoffs (interview-prep knowledge base)

**Purpose:** a study artifact, written retroactively after Phase 0 shipped (PR #1, merged
2026-06-09). Every non-trivial Phase-0 fork as **Problem → Options → Tradeoffs → Decision → How an
interviewer probes it.** Terse decision record: `docs/architecture-decisions.md`. Design:
`docs/design.md`. This is the "why," at interview depth. Companion: `docs/phase1-decisions-and-tradeoffs.md`.

**What Phase 0 is:** a cited Q&A assistant over Gen Academy course materials — member asks a
question → grounded, cited answer **or an honest refusal** — plus a 15-question evaluation report. The
overriding constraint was **"the graded artifact is the eval report, not a flashy demo"**: scope creep
toward UI was the #1 risk, so every decision below was made to protect a reproducible eval and a
load-bearing refusal path.

---

## Decision 1 — Hybrid retrieval (dense + BM25 + RRF) in Phase 0, not "dense now, hybrid later"

**Problem.** How does retrieval find the right chunks? Pure vector (dense) search is the default, but
it misses exact-keyword / rare-token matches (IDs, error codes, exact phrases).

**Options.** (a) dense-only (Chroma) · (b) hybrid: dense + BM25 sparse, fused · (c) dense + a
cross-encoder reranker.

**Tradeoffs.** Dense-only is simplest but loses on exact-match queries — and the eval deliberately
includes an `exact_match` category, so dense-only would *measurably* underperform. A cross-encoder
reranker is the strongest but adds a model + latency and is a bigger build. **Hybrid** captures both
signals (semantic + lexical) for one extra in-memory index and a fusion step — the best
quality-per-effort, and it matches the assignment's reference pattern ("hybrid + rerank").

**Decision: hybrid in Phase 0; rerank deferred to Phase 2.** Dense from Chroma (cosine), sparse from
`rank-bm25`, fused with **Reciprocal Rank Fusion**, `top_k=5`, `candidate_k=20`, `rrf_k=60`. Reranking
becomes a Phase-2 *before/after eval delta* (a deliberate demo moment), not a Phase-0 cost.

**Sub-decision: hand-rolled RRF, not LangChain's `EnsembleRetriever`.** RRF is ~5 lines
(`score = Σ 1/(k + rank + 1)`); rolling it keeps the fusion logic visible, testable, and free of a
framework abstraction we'd have to fight to inspect.

**How an interviewer probes it.**
- *"Why BM25 if you have embeddings?"* → dense search is weak on exact tokens (names, IDs, rare
  terms); BM25 is exact-lexical. Hybrid covers the failure mode of each. Cite the `exact_match` eval
  category as the concrete reason.
- *"Why RRF over score-weighted fusion?"* → RRF fuses *ranks*, not scores, so you don't have to
  normalize a cosine sim against a BM25 score (different scales, different distributions). Rank-based
  fusion is robust to that mismatch — that's its whole point.
- *"What's `k=60` in RRF?"* → a smoothing constant; larger `k` flattens the contribution of top
  ranks. 60 is the paper's default; it's not sensitive enough to tune in Phase 0.

---

## Decision 2 — Local embeddings, with the cloud model reserved for generation

**Problem.** Where do embeddings come from? The project must make at least one call to a model hosted
on Nebius (an assignment requirement).

**Options.** (a) Nebius/API embeddings · (b) local `sentence-transformers` embeddings + Nebius for
generation · (c) local for both (no cloud call — violates the requirement).

**Tradeoffs.** API embeddings add per-call latency, cost, rate limits, and a network dependency on the
*hot* path (every query and every chunk embeds). Local `all-MiniLM-L6-v2` is 384-dim, ~12 ms/query
after a one-time model load, deterministic, free, and offline — ideal for a reproducible eval. The
Nebius requirement is then satisfied where it adds the most value and the least fragility:
**generation** (one call per query, not per chunk).

**Decision: local sentence-transformers embeddings; Nebius = generation (the mandatory call).** This
also makes the embedding provider a clean Phase-2 swap ("swap to Nebius embeddings" demo) behind the
`ModelProvider` seam.

**How an interviewer probes it.**
- *"Why not embed with the same cloud model?"* → embeddings are on the hot path (N chunks at ingest,
  every query); putting that behind a rate-limited API hurts both eval reproducibility and latency.
  Generation is one call per query — the right place to spend the network dependency.
- *"384 dims — limitation?"* → smaller = faster/cheaper, slightly less expressive than 768/1024-dim
  models; fine for a focused corpus, and the seam lets you swap up later with a before/after eval.

---

## Decision 3 — Two-tier corpus: a commit-pinned eval set, immutable, walled off from uploads

**Problem.** The corpus *grows* (admin uploads, repos move). But a graded eval must be **reproducible**
— if the corpus shifts under it, recall@k is meaningless. And one source repo is the *sample solution*
to the assignment, which must never be ingested.

**Options.** (a) one mutable corpus · (b) two tiers: a commit-pinned **eval** corpus (graded) + a
growing **production/serving** corpus (uploads) · (c) snapshot the whole corpus per eval run.

**Tradeoffs.** One mutable corpus means every upload silently changes the eval baseline — fatal for a
graded, reproducible report. Per-run snapshots are heavy and still drift. **Two tiers** pin the eval
to exact commit SHAs (so line numbers and content are frozen) while letting production grow — at the
cost of maintaining two Chroma collections.

**Decision: two-tier.** **eval** = GitHub repos pinned to exact SHAs
(`awesome-agentic-ai-resources@5dfb869`, `Mastering-Agentic-AI-Week1@3aa31df`), graded, never written
to by the product. **serving** = the eval chunks seeded once + admin uploads, used by the live bot.
**Invariant: uploads only ever touch `serving`; `eval` is immutable** — enforced by a test that an
upload leaves `eval` empty/unchanged.

**Sub-decision: the Week-2 firewall.** The `Mastering-Agentic-AI-Week2` repo *is* the sample solution;
reading it is disqualifying. An allowlist (`assert_allowed`) raises on any repo not explicitly pinned,
so the sample solution is un-fetchable by construction, not by convention.

**How an interviewer probes it.**
- *"Why pin to a commit SHA?"* → eval reproducibility: a graded metric over a moving corpus is noise.
  The SHA freezes content *and* the line spans the gold answers reference.
- *"How do you keep uploads from polluting the benchmark?"* → physical separation (distinct
  collections) + an asserted invariant, not just discipline. The eval scripts read `eval`; the product
  reads/writes `serving`.
- *"Firewall — allowlist or denylist?"* → allowlist. A denylist of "don't read Week-2" fails open the
  moment a new repo appears; an allowlist fails closed.

---

## Decision 4 — Refusal grader: JSON-mode LLM primary + cosine-threshold fallback

**Problem.** The bot must refuse when the retrieved context doesn't support an answer ("I couldn't
find this in the course materials") instead of hallucinating from the model's priors. How is
"answerable?" decided?

**Options.** (a) the answer model self-reports confidence · (b) a pure cosine-similarity threshold ·
(c) a dedicated JSON-mode grader LLM call, with the threshold as a fallback.

**Tradeoffs.** Self-reported confidence is notoriously miscalibrated and conflates "I'm sure" with "the
context supports it." A pure threshold is deterministic and cheap but blunt (a high-similarity chunk
can still not *answer* the question). A **dedicated grader** call ("can this question be answered FROM
THIS CONTEXT ALONE?") is the most accurate, and a **cosine-threshold fallback** keeps the refusal path
working when the grader's JSON is unparseable or the API is down.

**Decision: JSON-mode grader LLM (primary) + cosine-threshold fallback (`0.2`).** The spike confirmed
JSON mode works on the open model, so the primary path is viable; the fallback makes refusal robust to
parse/availability failures. **The refusal path is load-bearing** — there is no path that emits an
answer when the grader says no.

**The bug this design later exposed (a fail-safe-parsing lesson).** The grader parsed the model's
`answerable` field with `bool(parsed["answerable"])`. JSON-mode models sometimes emit booleans as
**strings**: `{"answerable": "false"}`. In Python `bool("false") == True` — so a correct refusal was
silently flipped into an answer. Caught in the PR review. Fix: a `strict_bool` that accepts only real
`true`/`false` (bool or string) and **raises on anything else**, routing to the safe cosine fallback.
Parsing of a security-critical field must **fail safe** (toward refusal), never fall open.

**How an interviewer probes it.**
- *"Why not let the model just say 'I don't know'?"* → it conflates parametric confidence with
  context-groundedness and is poorly calibrated; a separate grader question ("answerable from THIS
  context?") is the signal you actually want.
- *"Your grader API dies — what happens?"* → the cosine-threshold fallback still makes a refuse/answer
  call; refusal never depends on a single fragile path.
- *"`bool('false')` is `True` — how do you defend against that?"* → strict parsing of security fields
  with fail-safe defaults; never coerce an arbitrary JSON value with `bool()`. (This is a real,
  shipped lesson, not hypothetical.)

---

## Decision 5 — Two evals, deliberately ranked: protected retrieval eval + cuttable faithfulness judge

**Problem.** The deliverable is "retrieval quality scores + where it fails and why." How much eval, and
what's protected if time runs out?

**Options.** (a) only deterministic retrieval metrics · (b) only LLM-as-judge faithfulness · (c) both,
explicitly ranked by what's graded.

**Tradeoffs.** Retrieval metrics (recall@k / precision@k / MRR) are **deterministic, LLM-free,
reproducible** — and they're literally what the handout grades. LLM-as-judge faithfulness is a richer
"is the answer grounded?" signal but is non-deterministic, rate-limited, and *not* the graded artifact.
Building only the judge risks shipping an impressive-but-unreproducible report; building only retrieval
misses a depth signal.

**Decision: both, ranked.** The **deterministic retrieval eval is the protected core** (never cut).
The **LLM-judge faithfulness is a cuttable depth add-on** — under rate-limit pressure it falls back to
a deterministic **citation-grounding check** (do the answer's content words appear in the retrieved
chunks?). The report labels which faithfulness scorer produced the number, so it's never misrepresented.

**Sub-decision: the gold set is 15 questions across 6 categories** (answerable, exact_match,
chunking_stress, multi_document, ambiguous, **unanswerable**), with an annotation gate (catalog rows
that merely *link* a resource aren't answerable "explain X" questions). The unanswerable + chunking +
multi-doc categories exist specifically to *stress* retrieval and the refusal path, not to inflate the
score.

**How an interviewer probes it.**
- *"Why deterministic metrics if you have an LLM judge?"* → reproducibility and grading: recall@k is
  the same every run and is the graded artifact; the judge is non-deterministic and supplementary.
- *"What's your fallback when the judge is rate-limited?"* → a deterministic citation-grounding check,
  and the report states which scorer was used — you never silently mix two scorers into one number.
- *"How do you design an eval set that's honest?"* → category balance including unanswerable +
  stress cases, and an annotation gate so 'linked' ≠ 'answerable'. An eval that only asks easy
  answerable questions measures nothing about refusal.

---

## Decision 6 — Fixed-size character chunking behind a `Chunker` interface

**Problem.** How is a document split into retrievable chunks? Chunking strategy is a real eval variable.

**Options.** (a) fixed-size character windows · (b) token-exact windows · (c) section/structure-aware
chunking.

**Tradeoffs.** Section-aware chunking is best for quality but is a bigger build and couples to each
format's structure. Token-exact needs the tokenizer on the path. **Fixed-size character windows**
(size 1000, overlap 150 ≈ ~250 tokens, safely under the embedder's 256-token cap) are simple,
format-agnostic, and good enough for a baseline — *and* because chunking is an eval variable, it lives
behind a `Chunker` Protocol so a `SectionAwareChunker` is a Phase-2 swap with a before/after delta.

**Decision: `FixedSizeChunker` (char-based) behind a `Chunker` interface; section-aware → Phase 2.**
Chunks carry exact char spans + 1-based line spans (GitHub) or page markers (PDF) for citations.

**How an interviewer probes it.**
- *"Char windows vs tokens?"* → chars are simple and tokenizer-free; the risk is overflowing the
  embedder's token cap, which is why the size is calibrated (~250 tok ≪ 256 cap). Tokens are more
  precise but add a dependency on the path.
- *"Why behind an interface if you only have one impl?"* → because it's a known eval axis; the
  assignment will compare chunking strategies, so the seam is earned, not speculative (contrast with
  the YAGNI call on the datastore split in Phase 1).
- *"Overlap — why 150?"* → to keep an answer that straddles a window boundary recoverable; the
  chunking-stress eval category exists to measure exactly where this still fails.

---

## Decision 7 — The `commit_hash` provenance gate (the chain that makes the eval honest)

**Problem.** A gold answer says "this fact is at `README.md` lines 12-14 of repo X **at commit Y**." A
retrieved chunk must be credited as a hit *only if it's genuinely that source* — not a same-path file
from a different commit, and never a production upload that happens to overlap.

**Options.** (a) match on repo + file + line overlap · (b) also require `commit_hash` equality.

**Tradeoffs.** Matching on path + lines alone lets content from a *different* commit (or a production
PDF) falsely satisfy a gold marker, inflating recall. Requiring `commit_hash` equality closes that —
at the cost of threading `commit_hash` unbroken through the whole pipeline (fetch → chunk metadata →
retrieved chunk → scorer).

**Decision: require `commit_hash` equality in `chunk_matches_span`.** The chain is unbroken by
construction; **production/uploaded chunks carry `commit_hash=None`, so they can never satisfy a gold
span** — the eval can only be passed by the pinned eval corpus. This is the data-integrity backbone of
the whole eval claim.

**How an interviewer probes it.**
- *"How do you know a retrieved chunk is really the gold source?"* → provenance equality, not just
  path/line overlap; the commit SHA is carried end-to-end and checked at scoring.
- *"Could an upload accidentally inflate recall?"* → no — uploads have `commit_hash=None` and the gate
  is equality against a real SHA, so production content structurally can't match gold.

---

## Decision 8 — Pure core / thin view + a deterministic offline test seam

**Problem.** This is a multi-layer app (retrieval, grading, web, eval). How do you keep it testable and
the pieces swappable, without every test needing API keys or a network?

**Options.** (a) let the web layer call models/stores directly · (b) pure core behind Protocol seams +
a fake provider for tests.

**Tradeoffs.** Direct coupling is faster to write but makes the core untestable offline and welds you
to FastAPI/Chroma/Nebius. **Protocol seams** (`ModelProvider`, `VectorStore`, `Retriever`, `Chunker`,
`Datastore`, `Loader`) + constructor injection cost a little upfront structure but buy: offline
deterministic tests, drop-in Phase-2 swaps (Pinecone, Postgres, reranker), and a core you can reason
about in isolation.

**Decision: pure core / thin view; no FastAPI/template imports in `core/` or `data/`.** A
`FakeModelProvider` (sha256-seeded 384-dim embed + scriptable generate) makes the entire suite run
offline and deterministically; the single live call is one `@pytest.mark.integration` test.

**Sub-decision: `RetrievedChunk.score` carries cosine similarity, NOT the RRF score.** This is subtle:
RRF decides *ranking*, but the grader's fallback needs a *similarity* signal. A pre-build review caught
that feeding the tiny RRF score (~0.02) to the cosine-threshold fallback would make it refuse *every*
query. Fix: `VectorStore.query` returns `(id, cosine_sim)`, carried as `score`; RRF stays
ranking-only.

**How an interviewer probes it.**
- *"How do you test a RAG pipeline without hitting an API on every run?"* → a deterministic fake
  behind the provider seam; real calls are isolated to one marked integration test. Determinism is
  what makes the eval and the tests trustworthy.
- *"Why Protocols instead of just classes?"* → they define the swap contract (Chroma→Pinecone,
  SQLite→Postgres) and let tests inject fakes; the seam is where Phase-2 plugs in.
- *"What broke when you conflated RRF score with similarity?"* → the refusal fallback saw ~0.02 ≪ 0.2
  threshold and refused everything; ranking score and confidence score are different signals and must
  be carried separately.

---

## Cross-cutting themes (good behavioral/architecture talking points)

1. **Protect the graded artifact, not the demo.** Every Phase-0 fork optimized for a reproducible eval
   and a load-bearing refusal path; UI polish was explicitly the first thing cuttable. Knowing what
   you'd cut *first* under time pressure is a maturity signal.
2. **Determinism is a feature.** Local embeddings, a fake provider, commit-pinned corpus, deterministic
   retrieval metrics — the whole spine is reproducible, which is what makes the eval *mean* something.
3. **Fail safe on security-critical paths.** The refusal grader fails toward refusal (strict bool parse
   → cosine fallback); production content can't satisfy gold (commit_hash gate). When in doubt, fail to
   the safe state.
4. **Earn your abstractions.** The `Chunker` seam is earned (chunking is a known eval axis); the
   datastore split is *not* yet (YAGNI, deferred to Phase 1/2). Same principle, opposite calls —
   the test is "is there a concrete second case or a known variable?"
5. **Carry provenance end-to-end.** `commit_hash` from fetch to scorer is what lets the eval claim be
   trusted; integrity comes from an unbroken chain, not a final check.

---

## A note on process (builder ≠ reviewer)

Phase 0's plan was reviewed by two other models before any code (Antigravity for code-fidelity,
Kimchi for design/eval methodology) — catching the RRF-vs-cosine score bug, a private-attribute
reach-in, and the upload-not-searchable-until-restart issue *in the plan*, before they were written.
After the build, a multi-agent PR review caught the `bool("false")` refusal-bypass and 9 other issues,
all fixed with regression tests before merge. The recurring lesson: **a second reviewer catches the
expensive class of bug (wrong data model, refusal bypass, integrity gap) cheaply — at plan/design time,
or at review time, not in production.**
