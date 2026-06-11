# GenAcademy RAG - Learnings So Far

**Date:** 2026-06-11 (originally 2026-06-09)
**Scope:** lessons from Phase 0, Phase 1, the first Phase 2 slice through PR #3, the deploy and
answer-trust slices, the Compass house-theme UI slice (PR #14), and the rerank re-enablement
analysis.
**Primary sources:** `docs/design.md`, `specs/roadmap.md`,
`docs/phase0-decisions-and-tradeoffs.md`, `docs/phase1-decisions-and-tradeoffs.md`,
`docs/phase2-decisions-and-tradeoffs.md`, `eval/REPORT.md`, `eval/phase2-rerank-delta.md`, and
`docs/deploy.md`.

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
   *Update 2026-06-11:* the posture flipped once the actual blocker was fixed. Baking the
   cross-encoder into the Docker image (PR #16) plus capping the pool (`GENACADEMY_RERANK_POOL=20`)
   kept the full recall win at lower latency, and rerank now ships **enabled** in the live Space.
   The config default stays `false` (deterministic local/eval baseline), with the env var as the
   no-rebuild kill switch.

5. **Chunking is still the next quality lever.**
   The current eval failures repeatedly point at table rows, section headers, and answers split across
   fixed-size chunk boundaries. Section-aware chunking is likely a better next Phase 2 slice than more
   retrieval tuning.
   *Update 2026-06-11:* the slice was built and measured — and lost (recall 0.67→0.64, MRR
   0.55→0.34; `eval/phase2-section-aware-chunking-delta.md`). Fixed chunking stays the default. The
   diagnosis still stands (q5/q7-class boundary misses persist), but heading-bounded chunks were the
   wrong remedy, partly confounded by the embedder's 256-token window truncating 1500-char chunks.

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

## Web UI And Templates

1. **Behavioral tests pin contiguous strings, so markup is a constrained surface.**
   The web tests assert exact substrings like `GenAcademy Compass` and
   `Evidence-first answers from the cohort materials.` Splitting a pinned phrase with an inline
   `<mark>` or wrapping a value in `<code>` broke four tests during the theme restyle — even a CSS
   *comment* containing the word "Sources" tripped the refusal test. Template changes must re-run
   `tests/web/` before claiming done.

2. **Separate skin from layout to keep template diffs reviewable.**
   The house theme lives as `gc-*` component classes in one `<style>` block in `base.html`;
   templates keep Tailwind utilities for layout only. A full visual restyle then touches class
   lists, not structure, and the pinned-string contracts mostly survive.

3. **Verify rendered output, not source.**
   Rendering real Jinja templates with sample contexts and screenshotting them headlessly caught
   visual problems (clipped labels, wrong palette fills) that reading the diff never would. The same
   loop worked for SVG diagrams, where CSS class fills silently override `fill` attributes.

4. **New server behavior ships with tests in the same slice.**
   The `/logout` route initially shipped with zero coverage in a suite that tests CSRF exhaustively
   everywhere else. Independent review caught it; six behavioral tests (session clearing, CSRF
   rejection, admin revocation, refusal recovery, XSS escaping) closed the gap before merge.

5. **A UI that hardcodes measured numbers will eventually lie.**
   The trust sidebar baked in `recall@k 0.67` and `refusal 0.73` — and the prediction came true
   within days: the rerank-enablement PR moved the frozen-eval refusal score to 1.00 and review had
   to catch the stale badge by hand. Measured values shown to users should be injected from the
   eval output, not typed into templates. (Still a follow-up; the badge is hand-synced for now.)

## Deployment And Latency Budgets

1. **Diagnose the actual blocker before designing around an assumed one.**
   Rerank was believed to be off in the Space "because of latency." The real blocker was that the
   cross-encoder model is not baked into the Docker image and `LOCAL_FILES_ONLY=true` blocks runtime
   downloads — a one-line Dockerfile fix. The latency budget (~6.7 s worst case vs an 8 s ceiling)
   actually fit. The proposed remedy (swap the generation model) was useful, but for a different
   reason than the one that motivated it.

2. **Itemize the latency budget before swapping components.**
   A request is embed (~12 ms) + retrieval (hundreds of ms; rerank multiplies it — see
   `eval/phase2-rerank-delta.md` for the measured runs) + grader LLM call (~0.5–1 s) + answer LLM
   call (~0.5–4 s). The two LLM calls dominate and vary the most; that is where freed budget comes
   from. Optimizing the wrong stage looks productive and changes nothing.

3. **Locally measured latency does not transfer to deploy hardware.**
   The committed 886 ms rerank p95 was measured on a development machine; the Space CPU is weaker
   and the figure could be 2–3×. Any latency-sensitive enablement needs a live measurement gate on
   the target hardware, plus a knob (`GENACADEMY_RERANK_POOL`) to shrink the work if it misses.

4. **An env-var kill switch beats a redeploy as a rollback plan.**
   `GENACADEMY_RERANK_ENABLED=false` restarts the app without a rebuild. Features that might blow a
   budget in production should be reversible at config speed, not image-build speed.

5. **Heavy compute inside a lock is a serialization decision, not just a latency one.**
   Rerank runs inside the corpus lock, so its cost serializes all concurrent asks. Acceptable at
   cohort traffic, but it should be a documented decision — the lock exists because dense and sparse
   retrieval are two consistency domains, and moving work out of it would reopen that problem.

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

6. **"Disabled due to latency" was the wrong story.**
   Rerank stayed off in the Space because the model was never provisioned into the image, not
   because the budget was blown. Re-checking the original reason before planning the fix changed
   the plan from "swap providers to afford rerank" to "bake the model in, cap the pool, and measure
   on the Space" — with the model swap kept as a quality/headroom improvement, not a prerequisite.

## Next Best Learning Target

Section-aware chunking was run as the next experiment — and measured as a regression
(`eval/phase2-section-aware-chunking-delta.md`: recall 0.67→0.64, MRR 0.55→0.34, partly confounded
by embedder tail-truncation at 1500 chars). The boundary failures it targeted (q5-class compact
table rows, multi-document spans outside top-5) remain the open quality problem. The
highest-signal next experiments, in order: re-run section chunking at `max_chars=1000` to remove
the truncation confound; adjacent-chunk stitching before generation; and `top_k`/query-decomposition
for the multi-document misses — each with the same before/after eval-delta discipline.
