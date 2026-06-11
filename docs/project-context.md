# GenAcademy RAG — Project Context

**Purpose of this file:** a single accurate primer so an AI assistant (or a new contributor) can
answer questions about this repo without re-reading the whole codebase. Verified against the source
on **2026-06-11** (commit `ec90bca`, after PR #16 merged). If code and this file disagree, the code
wins — and this file should be updated.

## What this is

A **grounded, cited Q&A assistant over Gen Academy cohort materials** ("what did the course actually
say about X?"). Members ask questions and get a **cited answer or an honest refusal** — never an
answer from model priors. Admins manage the corpus and monitor usage. Built as the Week-2 cohort
project ("Grounding AI with RAG & Context Engineering") and as a **job-search portfolio piece**.

- User-facing app title: **GenAcademy Compass** (repo/package keep the `genacademy-rag` name)
- Live app: <https://Manjunath84-genacademy-rag.hf.space> (Docker Hugging Face Space)
- GitHub: <https://github.com/manjunath84/genacademy-rag>
- Explicitly separate from the sister project `legal-rag-private` (no privacy/on-prem thesis here)

## Tech stack (pinned in `pyproject.toml`)

Python ≥3.12, managed by **uv**. FastAPI 0.136 + Jinja2 + Starlette sessions (server-rendered, HTMX
for feedback, Alpine.js for small UI state; **no SPA**). LangGraph 1.2.4 (one small graph).
chromadb 1.5.9 (local persistent) and pinecone 9.1.0 (serverless) behind one seam.
sentence-transformers 5.5.1 (local embeddings + cross-encoder rerank). openai SDK 2.41 (used as the
generic OpenAI-compatible client for every generation provider). rank-bm25, pypdf, nbformat, bcrypt,
pyyaml. Dev: pytest 9 (266 passed, 1 deselected `integration` marker as of 2026-06-11), ruff 0.15
(line-length 100; `spike/` excluded from lint).

Standard commands:

```bash
uv run ruff check .
uv run pytest -q                                # integration tests auto-deselected
uv run python scripts/ingest_eval_corpus.py     # seed the pinned eval corpus (local Chroma)
uv run python scripts/eval_retrieval.py --collection eval   # deterministic eval, no LLM key needed
uv run python scripts/run_eval.py               # full 15-q eval incl. LLM judge (spends tokens)
uv run uvicorn genacademy_rag.web.main:app --port 7860      # local app
```

## Architecture in one paragraph

**Pure core / thin view** (a `from fastapi import` inside `src/genacademy_rag/core/` is a review
reject). All seams are Protocol + config preset, not branching: `ModelProvider`, `VectorStore`,
`Retriever`, `Chunker`, `Reranker`, `Datastore`. Everything is wired from env in
`web/app.py:build_default_app()`; `create_app(retriever, provider, datastore, ...)` takes injected
fakes so the whole web layer is testable offline. The request path is one LangGraph graph:
**retrieve → grade → {answer + citations | refuse}**.

## Repo layout

```
src/genacademy_rag/
  config.py                 Settings.from_env(); PROVIDER_PRESETS (nebius/openrouter/openai/gemma)
  core/
    types.py                Citation, Chunk, Document, RetrievedChunk, GraphState (all frozen)
    chunker.py              FixedSizeChunker (default), SectionAwareChunker; build_chunker()
    providers.py            STEmbedder (local MiniLM), OpenAICompatEmbedder, OpenAICompatProvider,
                            CombinedProvider; build_provider()/build_embedder()
    vectorstore.py          ChromaStore, PineconeStore; build_vectorstore()
    retriever.py            HybridRetriever: dense + BM25 → RRF → optional rerank; corpus lock
    reranker.py             SentenceTransformersCrossEncoderReranker; build_reranker()
    grader.py               JSON-mode LLM answerability grade; cosine-threshold fallback
    graph.py                the LangGraph graph; REFUSAL_MESSAGE; answer prompt
    pipeline.py             IngestPipeline (prepare/commit w/ rollback), QueryPipeline → QueryResult
    sources.py              merge_citations() → SourceView rows; confidence_bucket(); github_url()
    security.py             bcrypt hash/verify; invite codes as id.secret pairs
    json_utils.py           strict_bool() — "false" string must not become True
    analytics.py            usage_summary() for the admin dashboard (p50/p95, refusal/fallback rate)
    loaders/                EVAL_CORPUS allowlist + assert_allowed() firewall; github/markdown/
                            jupyter/pdf loaders
  data/datastore.py         SQLiteDatastore: users, documents, chunks_meta, invite_codes,
                            usage_log, feedback (schema at top of file)
  deploy/bootstrap.py       first-boot: seed/verify the pinned eval Chroma collection
  eval/                     gold_schema.py, retrieval_eval.py, faithfulness_eval.py, report.py,
                            gold/gold_set.yaml (15 questions)
  web/
    app.py                  all routes; create_app() + build_default_app() (real wiring)
    auth.py                 authenticate() (delegates to core security)
    main.py                 ASGI entrypoint (build_default_app())
    templates/              base.html (gc-* component CSS), chat.html, login/signup,
                            admin_{invites,documents,dashboard}.html
scripts/                    ingest_eval_corpus, eval_retrieval, run_eval, provision_rerank_model,
                            smoke_http, smoke_pinecone, smoke_nebius_embeddings, start_hf_space.sh
tests/                      mirrors src; tests/web/test_app.py pins exact UI strings (see Contracts)
eval/REPORT.md              committed eval report; eval/phase2-*-delta.md = A/B experiment records
specs/                      mission.md, roadmap.md (phases/MUST-vs-SHOULD), tech-stack.md
docs/                       design, decisions-and-tradeoffs per phase, deploy.md, project-writeup.md,
                            learnings.md, demo-script.md, diagrams/
AGENTS.md                   binding working agreement (CLAUDE.md mirrors it)
```

## Query path (what happens on POST /ask)

1. **Embed** the question — local `all-MiniLM-L6-v2`, 384-dim, normalized (~12 ms; model loaded
   once at boot, cold ~11.6 s).
2. **Dense search** — `VectorStore.query()` returns `(chunk_id, cosine similarity)`;
   `candidate_k=20`. Chroma reports distance → converted `1 - d`; Pinecone reports similarity
   directly (contract documented in `vectorstore.py`).
3. **Sparse search** — BM25 (rank-bm25) over an in-memory index of the whole corpus, top 20.
4. **RRF fusion** — `rrf_fuse()` with k=60 produces the ranking.
5. **Optional rerank** — if `GENACADEMY_RERANK_ENABLED`, cross-encoder
   `cross-encoder/ms-marco-MiniLM-L6-v2` rescores the fused union (capped at
   `GENACADEMY_RERANK_POOL`; 0 = full union) and reorders. Runs **inside the corpus lock**
   (deliberate: the lock spans two consistency domains — Chroma + BM25 snapshot — so rerank cost
   serializes concurrent asks; accepted at cohort traffic).
6. **Truncate to top_k=5.** Each `RetrievedChunk.score` is the **cosine similarity** (0.0 for
   BM25-only hits) — *not* the RRF or rerank score. This is a load-bearing contract: the grader
   fallback reads it.
7. **Grade** (`grader.py`) — JSON-mode LLM call (max_tokens=64): "answerable from THIS context
   alone?". Parsed with `strict_bool` (a stringified `"false"` must not become `True`). On any
   parse/provider failure → **cosine fallback**: max retrieved score ≥ 0.2 threshold, flagged
   `used_fallback=True` and logged (a degraded grader must not look healthy).
8. **Answer or refuse** (`graph.py`) — answerable → second LLM call (max_tokens=800; overview
   paragraph + bullets, context-only). Not answerable → fixed refusal
   `"I could not find this in the course materials."` Citations ride along either way;
   `QueryPipeline.answer()` indexes graph output keys directly (missing key = wiring bug = loud
   KeyError).
9. **Render** — `sources.merge_citations()` dedupes per-chunk citations into `SourceView` rows with
   pinned-commit GitHub URLs (`https://github.com/The-Gen-Academy/<repo>/blob/<sha>/<path>#L..`) or
   `/documents/{doc_id}/file` for uploads. Confidence 1–5 → low/medium/high badge. Refusals show
   recovery suggestion pills, no source list. The query is logged to `usage_log` (latency_ms,
   refused, used_fallback, n_citations); log failure never breaks the answer.

Two LLM calls per answered question (grader + answer), one for refusals. Latency budget: < 8 s hard
ceiling (handout rule), ~6 s goal.

## Ingest and the two-tier corpus

- **`eval` collection (frozen, always local Chroma):** the commit-pinned GitHub corpus defined in
  `core/loaders/__init__.py:EVAL_CORPUS` — `The-Gen-Academy/awesome-agentic-ai-resources@5dfb869`
  (README) and `Mastering-Agentic-AI-Week1@3aa31df` (notebook + README + a .py treated as text).
  `assert_allowed()` is the **firewall**: only those exact (owner, repo, sha) triples are fetchable;
  `Mastering-Agentic-AI-Week2` (the sample solution) is absent by construction and reading it is
  disqualifying. `ingest_eval_corpus.py` refuses non-fixed chunkers / non-local embeddings for the
  `eval` collection.
- **`serving` collection (grows with uploads):** swappable via `GENACADEMY_VECTORSTORE`
  (chroma|pinecone). Seeded once from the eval chunks if empty. Admin PDF uploads go through
  `IngestPipeline` (vector store first, then SQLite ledger; on ledger failure the vectors are rolled
  back). Upload/delete/reindex mutate the retriever corpus under the lock; deletes rebuild from the
  in-memory snapshot (never a remote re-read — Pinecone is eventually consistent); **reindex is the
  one deliberate remote re-read**, filtered against the SQLite ledger so orphaned/deleted uploads
  can't resurrect (matters because HF `/data` resets while Pinecone persists).
- Chunking default: **fixed 1000 chars / 150 overlap** (`GENACADEMY_CHUNKER=fixed`).
  `SectionAwareChunker` (markdown-heading-bounded, max 1500 chars) exists but **measured as a loss**
  (recall 0.67→0.64, MRR 0.55→0.34 — see `eval/phase2-section-aware-chunking-delta.md`; confound:
  1500-char chunks exceed the embedder's 256-token window). Fixed remains the default.
- Citations are captured **at ingest, never reconstructed**: every chunk carries doc_id/title/
  source_type plus repo/file_path/commit_hash/line span (GitHub) or page/char span (files), flattened
  into vector metadata and threaded untouched to the answer card and the eval scorer.

## Configuration (all env-driven; `config.py`)

| Variable | Default | Notes |
|---|---|---|
| `GENACADEMY_PROVIDER` | `openrouter` | preset: nebius / openrouter / openai / gemma (local :8085) |
| `NEBIUS_API_KEY` / `NEBIUS_MODEL` / `NEBIUS_BASE_URL` | — / — / tokenfactory URL | per-preset triples exist for each provider |
| `GENACADEMY_EMBEDDINGS` | `local` | local (MiniLM 384-dim) or nebius (Qwen3-Embedding-8B) |
| `GENACADEMY_EMBED_MODEL` / `GENACADEMY_EMBED_DIM` | `all-MiniLM-L6-v2` / 384 | Pinecone index dim must match |
| `GENACADEMY_VECTORSTORE` | `chroma` | chroma or pinecone (serving collection only) |
| `PINECONE_API_KEY`, `GENACADEMY_PINECONE_INDEX/CLOUD/REGION` | — / genacademy-rag / aws / us-east-1 | |
| `GENACADEMY_TOP_K` | 5 | |
| `GENACADEMY_CHUNKER` | `fixed` | fixed or section |
| `GENACADEMY_CHUNK_SIZE` / `GENACADEMY_CHUNK_OVERLAP` | 1000 / 150 | fixed chunker |
| `GENACADEMY_SECTION_CHUNK_MAX_CHARS` / `_OVERLAP` | 1500 / 150 | section chunker |
| `GENACADEMY_RERANK_ENABLED` | `false` | the rerank kill switch — flips without rebuild |
| `GENACADEMY_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | baked into the Docker image |
| `GENACADEMY_RERANK_POOL` | 0 (= full fused union) | 20 is the validated serving cap |
| `GENACADEMY_RERANK_LOCAL_FILES_ONLY` | `true` | blocks runtime downloads; provision script is the only download path |
| `GENACADEMY_RERANK_BATCH_SIZE` / `_DEVICE` / `_CACHE_DIR` | 32 / None / None | |
| `GENACADEMY_SESSION_SECRET` | `dev-only-change-me` | warns in dev; **hard-fails** if secure_cookies=true |
| `GENACADEMY_SECURE_COOKIES` | `false` | Docker image sets `true` |
| `GENACADEMY_DATA_DIR` / `GENACADEMY_CHROMA_DIR` / `GENACADEMY_SQLITE` | `./data` / data/chroma / data/genacademy.sqlite | image sets DATA_DIR=/data |

Boolean parsing is strict (`_env_bool`): an unrecognized value **raises** rather than silently
coercing a typo to False.

## Web layer

Routes (all in `web/app.py`; session auth, CSRF token in session compared via
`hmac.compare_digest` on every mutating POST):

| Route | Access | Notes |
|---|---|---|
| GET/POST `/login`, POST `/logout`, GET/POST `/signup` | public | signup redeems invite codes |
| GET `/` , POST `/ask` | any logged-in user | chat page; logs to usage_log |
| POST `/feedback` | logged-in, own query only | HTMX swap; verdict ±1, upsert per (query,user) |
| GET `/documents/{doc_id}/file` | logged-in | serves uploaded originals (inline PDF) |
| POST `/upload`, GET `/admin/documents`, POST `/admin/documents/{delete,reindex}` | admin | uploads are PDF; filename sanitized to basename |
| GET/POST `/admin/invites`, POST `/admin/invites/{id}/revoke` | admin | invite shown once |
| GET `/admin/dashboard` | admin | usage_summary + feedback counts |

- **Auth:** seeded users `admin@genacademy.local`/`admin` and `member@genacademy.local`/`member`
  (bcrypt-hashed; legacy plaintext rows migrated on boot). RBAC checked per-route via
  `require_admin()`, not hidden nav.
- **Invite codes** are bearer credentials shaped `id.secret`: id is the lookup key, secret is
  bcrypt-verified (bcrypt hashes aren't lookupable). Single-use, expirable, revocable; redemption is
  a `BEGIN IMMEDIATE` transaction.
- **UI test contract (important):** `tests/web/test_app.py` asserts exact contiguous substrings
  (e.g. `GenAcademy Compass`, `Evidence-first answers from the cohort materials.`, refusal pages
  must not contain the word "Sources" anywhere — including CSS comments). **Never split a pinned
  phrase with inline tags.** Theme lives as `gc-*` component classes in one `<style>` block in
  `base.html`; Tailwind utilities are for layout only.

## Eval system

- **Gold set:** 15 questions in `eval/gold/gold_set.yaml` — categories: answerable, exact_match,
  chunking_stress, multi_document, ambiguous, unanswerable (3 unanswerable; 12 retrieval-scored).
  Gold spans pin repo+file+**commit_hash**+line range; a retrieved chunk only counts if provenance
  matches (production content can never satisfy a gold marker). Schema invariants enforced at
  construction (`gold_schema.py`).
- **Deterministic retrieval eval** (the protected, handout-graded artifact): recall@k, precision@k,
  MRR. `scripts/eval_retrieval.py` needs no LLM key. Small-N caveat: n=12, one question moves
  aggregates by ~0.08.
- **Refusal correctness:** refused XOR answerable, over all 15.
- **Faithfulness** (depth add-on, cuttable): LLM-as-judge (pinned prompt, temp 0, raw outputs saved
  to `eval/runs/`); falls back run-wide to a deterministic citation-grounding word-overlap check on
  the first judge failure. **Known caveat (disclosed in the reports): the judge is the same
  provider/model as the generator** (`scripts/run_eval.py` builds one provider) — so faithfulness
  numbers self-grade and are not comparable across generation-model changes.
- **Current committed numbers** (`eval/REPORT.md`, 2026-06-11, Nebius
  `Qwen/Qwen3-30B-A3B-Instruct-2507`, rerank on, pool=20): recall@k **0.79**, precision@k **0.25**,
  MRR **0.58**, refusal correctness **1.00**, faithfulness 100% (self-judged — see caveat).
  Rerank delta vs baseline hybrid (0.67/0.22/0.55) is recorded in `eval/phase2-rerank-delta.md`.
  Remaining known misses: q5 (exact-match table row not retrieved), q9/q10 (multi-document second
  span outside top-5), q12 (ambiguous, partial).
- `scripts/run_eval.py` **overwrites `eval/REPORT.md` wholesale**; the failure-analysis narrative is
  hand-authored on top — re-add it after regeneration (known rot trap).

## Deployment (Hugging Face Space, Docker)

- `Dockerfile`: uv image, non-root `user`, `GENACADEMY_SECURE_COOKIES=true`,
  `HF_HOME=/app/.cache/huggingface`. **Both models are baked in at build time** — the MiniLM embedder
  and the cross-encoder rerank model (via `scripts/provision_rerank_model.py`, run with a build-only
  dummy session secret because `Settings.__post_init__` rejects the default secret under secure
  cookies). Port 7860.
- Boot (`scripts/start_hf_space.sh`): `deploy/bootstrap.py` seeds/verifies the pinned eval Chroma
  collection (re-fetches + re-embeds on a cold `/data`), then uvicorn **single worker** (in-process
  BM25 snapshot + SQLite are not multi-process safe).
- Live config: Nebius generation, **Pinecone serving store** (`GENACADEMY_VECTORSTORE=pinecone`),
  local embeddings, eval pinned to local Chroma regardless. Full variable list + 9-step Live
  Acceptance-Test Order in `docs/deploy.md`.
- HF `/data` is ephemeral without paid persistence: SQLite/uploads/Chroma reset on restart; Pinecone
  persists independently (hence the ledger-filtered reindex).
- **Rollback for rerank/model changes is env-var speed:** flip `GENACADEMY_RERANK_ENABLED=false` or
  revert `NEBIUS_MODEL` in Space variables — restart, no rebuild.

## Binding project rules (AGENTS.md — review-blockers)

1. Branch off `main`, never commit directly to it (PRs only).
2. **Builder ≠ reviewer:** a different model/fresh context reviews every non-trivial change.
3. **Evidence before "done":** show ruff + pytest output (and eval tables / live runs for behavior).
4. Pure core / thin view; citations at ingest; **the refusal path is load-bearing** — never weaken it
   to make a demo look smarter; pluggability via interface+config, not branching.
5. Never quote unverified numbers; API shapes/model IDs copied verbatim, not from memory.
6. **Never ingest/replicate `Mastering-Agentic-AI-Week2`** (sample solution) — firewalled in code.
7. Regenerated eval reports must run on the **Nebius preset** (mandatory-call mandate).

## Status as of 2026-06-11

Phases 0–2 shipped: graded spine + eval (Phase 0), RBAC/invites/uploads/dashboard/feedback (Phase 1),
rerank + Pinecone + Nebius-embeddings presets, section-chunking experiment (negative result),
Compass UI theme, HF Space deploy (Phase 2). PR #16 (bake rerank model into image, switch to
Qwen3-30B-A3B-Instruct-2507, rerank pool=20 eval) is **merged**; review findings addressed in
`969f52a`.

**Live Space configuration (since 2026-06-11):** `NEBIUS_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507`,
`GENACADEMY_RERANK_ENABLED=true`, `GENACADEMY_RERANK_POOL=20`, Pinecone serving store, local
embeddings. Space variables flipped and live validation done by the owner on 2026-06-11.
Outstanding: demo video (≤5 min, script in `docs/demo-script.md`) and cohort form submission;
possible later improvements (e.g. injecting eval metrics into the UI instead of hardcoding,
chunking follow-ups per `docs/learnings.md` "Next Best Learning Target").

## Doc map (what to read for what)

| Question | Read |
|---|---|
| Why this stack / what was rejected | `docs/architecture-decisions.md`, `docs/design.md` |
| Phase-by-phase decisions + tradeoffs | `docs/phase{0,1,2}-decisions-and-tradeoffs.md`, `docs/answer-ux-decisions-and-tradeoffs.md` |
| Scope, MUST vs SHOULD, cut order | `specs/roadmap.md`, `specs/mission.md` |
| Current eval numbers + experiment deltas | `eval/REPORT.md`, `eval/phase2-rerank-delta.md`, `eval/phase2-section-aware-chunking-delta.md` |
| Deploy/runbook | `docs/deploy.md` |
| Distilled lessons | `docs/learnings.md` |
| Submission narrative | `docs/project-writeup.md` |
| Measured latency/API probes | `docs/spike-findings.md`, `spike/` (frozen, unlinted) |
