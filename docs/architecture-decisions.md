# GenAcademy RAG — Architecture Decisions

*Captured during the Week-2 brainstorming session (Gen Academy, "Grounding AI with RAG & Context
Engineering"). This is the decision log + reasoning. The full design spec and the executable plan
are separate documents; this is the "why" behind the locked choices so a future session (or a
hiring manager reading the repo) can see the thinking, not just the result.*

**Date:** 2026-06-07 · **Status:** stack locked, design in progress.

---

## 1. What this is

A **knowledge assistant for Gen Academy cohort members**: ask any question about the cohort's
materials (lecture decks, handouts, glossary, guidebooks, chat-Q&A docs) and get a **cited** answer.
**Admins** upload and manage the corpus and monitor usage; **members** only chat. Multi-format
ingestion (PDF, DOCX, TXT, web pages). Architected so **data sources, model providers, and retrieval
strategies are swappable**.

Maps to the handout's **Use Case #1 (Enterprise Policy Q&A Bot)** + the bonus **Chatbot UI** add-on,
**Track 2 (code-heavy: LangChain + LangGraph)**.

## 2. Relationship to `legal-rag-private` — DELIBERATELY SEPARATE

This is a **fresh standalone build (Branch B)**, its own repo, *not* built on the `raglab` core.

- `legal-rag-private` stays the **regulated-docs / privacy-thesis** portfolio piece (local Gemma,
  on-prem, "documents never leave your infrastructure"). Don't bolt that narrative onto this.
- This is a **different** portfolio artifact: a **multi-user internal knowledge product** (RBAC,
  admin dashboard, usage analytics, cloud-deployable). Two distinct strengths to a hiring manager,
  not the same idea twice.
- The privacy thesis **does not apply here** and should not be claimed — this runs over non-sensitive
  course materials and routes calls to a cloud API (Nebius).

## 3. Locked decisions

| Decision | Choice | Why (and what we rejected) |
|---|---|---|
| **Project relationship** | Fresh standalone build | Keeps legal-rag's privacy thesis clean; yields a second, *different* portfolio artifact. Rejected: build-on-raglab (couples two products), generalize-raglab-into-engine (most upfront refactor). |
| **Deployment posture** | **Hybrid: local-first, deploy-ready** | Fully working locally for the demo + a clean deploy path (Docker / HF Space), actually deploy only if time permits. Best one-week risk/reward. Rejected: genuinely-deployed (infra risk), local-demo-only (smallest wow). |
| **Build track** | Track 2 (code: LangChain + LangGraph) | Real RBAC, polished UX, pluggable Python core — all beyond n8n's ceiling. |
| **Stack / UI** | **FastAPI + HTMX + Tailwind + Alpine.js** | Single Python service, server-session auth, no JS build step, one Docker image — spend the complexity budget on RAG, not React. See §4.1. |
| **RAG orchestration** | LangChain primitives + **one** small LangGraph graph (refusal/escalation branch only) | Linear retrieve→rerank→generate is LCEL; LangGraph earns its place only for the confidence→{answer\|refuse} state machine. See §4.2. |
| **Model provider** | **Nebius Token Factory** (generation; embeddings optional) | Mandatory cohort constraint *and* cloud API → identical local/deployed, no local GPU. Free credits expected. See §4.3. |
| **Vector store** | **Pluggable: Chroma (local) ↔ Pinecone (cloud)** | Two config presets behind one `VectorStore` interface = the "swappable" requirement made concrete + a great demo moment. Free Pinecone credits expected. |
| **Relational DB** | **Pluggable: SQLite (local) ↔ Postgres (deploy)** | Pinecone holds vectors only; users/roles, document metadata, and the **usage log** need a relational store. The usage table *is* the admin-dashboard differentiator. See §4.5. |
| **Auth** | Server-session + role (`admin` / `member`) — *details TBD in design* | Trivial with HTMX (server sessions); avoids JWT/CORS tax of an SPA. Exact signup/gating model still open. |

## 4. The reasoning (architect's notes)

### 4.1 HTMX/Tailwind vs React
Grade and Builder-of-the-Week value come from retrieval quality, citations, the refusal path, and
the admin/usage story — **not** frontend sophistication. React spends the complexity budget in the
wrong place: JWT auth + CORS + two codebases + a build pipeline. HTMX gives server-session auth
(~20 lines), one container, and CRUD-by-fragment for the admin dashboard for free. The *one* place
React genuinely wins is the chat surface (streaming, citation popovers) — but HTMX's **SSE
extension** + Alpine deliver a *good* streaming chat at this scope. The delta isn't worth a week's
tax. **Flip to React only if** the chat needs rich multi-pane interactivity *and* React muscle
already exists. Not this scope.

### 4.2 Why only one LangGraph graph
LangGraph earns its keep on **branching state**. The handout's headline quality bar — *"design the
refusal first"* + confidence-based escalation — is a real state machine:
`retrieve → grade confidence → {high: answer + citations | low: refuse / escalate}`. That single
branch is a legitimate, demonstrable graph and the exact substrate **Week 3 (Agentic Leap)** reuses.
Everything else stays LCEL. If time-squeezed, the refusal collapses to a Python `if score <
threshold` and LangGraph drops with zero loss of correctness — it's a depth signal, not load-bearing.

### 4.3 Nebius constraint, in practice
"At least one model call through Nebius" = point the OpenAI SDK `base_url` at Nebius Token Factory's
OpenAI-compatible endpoint for embeddings or generation (or both). Recommended: **generation** on
Nebius (Llama/Qwen/DeepSeek-class open model); embeddings on Nebius too (simplest) or local
`sentence-transformers`. Because it's a cloud API it behaves identically local vs deployed — this is
what makes the deploy-ready half *cheap* (no GPU on the box), unlike legal-rag's laptop-welded Gemma.
**Bonus:** Nebius is a real inference platform, so unlike the local `mlx_vlm.server` it very likely
supports **JSON mode / structured output** → a cleaner confidence-grader than legal-rag's hand-parsed
yes/no. **Action:** 10-min capability spike before locking the grader design.

### 4.4 "Pluggable" is orthogonal to the stack
The `ModelProvider` / `VectorStore` / `Retriever` interfaces + a config-driven factory + two presets
live entirely in the **Python core**. The frontend calls a thin service layer and never knows which
backend is active. So pluggability did **not** influence the UI choice — stack was decided on
UX/auth/time grounds alone.

### 4.5 Two pieces the feature list implied but didn't name
- **Relational DB alongside Pinecone** for users/roles, document metadata, and the usage log.
  Pinecone stores vectors only. The usage table powers "monitoring usage" — the admin differentiator.
- **Citations are a data-model decision, not a UI one.** At ingest, stamp every chunk with
  `{doc_id, title, page/section, char_span}`, carry it through retrieval, return it with the answer,
  render as expandable source cards. If metadata isn't captured at ingest, no frontend can show
  citations — so it's designed into the core from day one.

## 5. Consolidated stack (locked)

```
FastAPI (one service)
├── HTMX + Tailwind + Alpine.js         (server-rendered UI; non-streaming form-post baseline, SSE optional)
├── Auth: server sessions + role gate   (admin / member)
├── RAG core (pure, testable)
│   ├── ingestion: pluggable loaders    (PDF / DOCX / TXT / web)
│   ├── chunk + embed                   (embeddings via Nebius or local ST)
│   ├── VectorStore  iface ── Chroma (local) ↔ Pinecone (cloud)
│   ├── Retriever    iface ── hybrid (dense + BM25, Phase 0; rerank = Phase 2)
│   └── LangGraph: retrieve → grade → {answer+citations | refuse/escalate}
├── ModelProvider iface ── Nebius (generation; mandatory call)
└── Relational DB iface ── SQLite (local) ↔ Postgres (deploy)
    └── tables: users, documents, chunks_meta, usage_log
```

## 6. Open questions (resolved in design.md §8)

- [x] **Auth/gating model** — Google OAuth on cohort emails vs invite-code signup vs seeded
      username/password. Pick one.
- [x] **MVP vs later decomposition** — what ships in the one-week core vs stretch goals.
- [x] **Embeddings provider** — Nebius vs local `sentence-transformers` (cost/latency/capability).
- [x] **Eval plan** — handout wants a 15-question evaluation report (faithfulness + failure
      analysis); design the question set + metric early.
- [x] **Ingestion processing** — synchronous-with-progress (MVP) vs background worker (stretch).

---

---

## 7. Review resolutions — Kimchi independent review (2026-06-07)

*Learnings log: the design was pressure-tested by an independent reviewer (`docs/design-review.md`)
before any code (the builder≠reviewer gate). Verdict: "good architectural taste, NOT yet plan-ready"
until three blocking blanks were closed. Every finding and the decision taken, for later review:*

| # | Kimchi's finding | Decision taken |
|---|---|---|
| **1.1** | Hybrid (BM25) belongs in Phase 0 — Use Case #1 literally says "Hybrid + rerank"; dense-only is a grading liability; `rank-bm25` is ~30 min | **Accept** — promote minimal dense+BM25+RRF to Phase 0; keep cross-encoder rerank in Phase 2. *No conflict with the earlier "finish first" advisor note — that warned against **product** scope creep, not a 30-min graded-retrieval win.* |
| **1.2** | Confidence-grader mechanism is a blank spot | **Accept** — LLM grader via Nebius JSON mode (pending spike), cosine-similarity threshold as fallback. |
| **1.3** | 20 MB guidebook has no parse fallback | **Accept** — parse-quality gate → OCR fallback → exclude-if-bad, in the spike. |
| **2.1** | Only 1 unanswerable Q is not a stress test | **Accept** — ≥3 unanswerable + chunking-stress questions. |
| **2.2** | Faithfulness scorer is "and/or" — pick one | **Accept** — LLM-as-judge, pinned rubric, temp=0, raw outputs saved. |
| **2.3 / 4.5** | Close chunking on fixed-size for Phase 0 | **Accept** — fixed-size (512/64) Phase 0; section-aware = Phase 2 eval axis; **add a `Chunker` interface**. |
| **2.4** | Declare the cut order | **Accept** — SSE→cards→admin-upload→(never) eval/refusal; Phase 0 UI baseline becomes **non-streaming form-post**, SSE optional. |
| **4.5 (embeddings)** | Embeddings local ST, Nebius = generation only | **Accept** — local `all-MiniLM-L6-v2` (384-dim) Phase 0; Nebius generation is the mandatory call; Nebius embeddings = Phase 2 swap demo. |
| **4.2 / 4.5** | Add precision@k + MRR, failure taxonomy, top-k=5 | **Accept.** |
| **3.1 / 3.3 / 3.4 / 4.6** | Annotation time (~6 h), move `usage_log`→Phase 1, spike rate-limits, pin LangChain versions, "eval green by Day 2 before UI polish" | **Accept all.** |
| **3.2** | "Divergences from sample solution" in the write-up | **Accept.** |
| **4.4 (architecture)** | Add `Chunker` interface; watch `Datastore` God-interface for Postgres | **Accept** — `Chunker` added; split `Datastore` into User/Doc/Usage stores when Postgres preset arrives. |

**Net effect:** the three blocking blanks (grader mechanism, hybrid-in-Phase-0, PDF parse fallback)
are closed in `design.md`; the spike (§9) now gates more (JSON mode, throughput, parse quality).
Design is **plan-ready pending the spike.** Meta-learning: the builder≠reviewer gate paid for itself —
the independent pass caught a use-case/pattern mismatch (dense-only vs the handout's prescribed hybrid)
that the build session had rationalized as "already gradeable."

**Follow-up refinement to 2.2 (faithfulness scorer), 2026-06-07 — handout re-read.** Re-checking the
handout deliverable (*"15-question report with **retrieval quality scores** and **failure analysis**"*)
showed it grades a **retrieval** eval, not generation faithfulness. So the two evals are now **ranked,
not just "separate":** the deterministic recall/precision/MRR + failure-taxonomy eval is the
**handout-mandatory spine (never cut)**; the LLM-as-judge faithfulness layer is a **self-imposed depth
add-on** that costs ~30 Nebius calls against a rate-limited free tier and is therefore **cuttable**,
falling back to a deterministic **citation-grounding check** (cited spans must appear in retrieved
chunks). Cut order updated in `AGENTS.md` / `roadmap.md`. Meta-learning: "separate evals" wasn't enough —
under a Day-2 deadline you must also know *which* eval is load-bearing for the grade, or rate-limit
pain silently jeopardizes the graded artifact for an ungraded one.

---

## 8. Antigravity plan — pre-review; its substance, reconciled (2026-06-07)

Antigravity produced two planning docs (`agy_week2-project-advisory.md`, `agy_implementation_plan.md`),
**both written against *this* decisions doc only — NOT against `design.md`, `design-review.md`, or the §7
review resolutions above.** On the big architectural calls they reflected the **pre-Kimchi** stack and
**regressed** on four points the review already corrected. The files have since been **deleted** (their
useful substance — the regressions to avoid, the corrections to absorb, the open trivia — is preserved in
this section; their scaffold shape is logged in memory). **If a planning agent revives that scaffold, do
not use it as-is** — reconcile to the reviewed `design.md` first, or it will re-bake the regressions below.

**AGY regressions (we HOLD the reviewed position):**

| Topic | AGY (pre-review) | Reviewed position (holds) |
|---|---|---|
| Hybrid retrieval | dense-only MVP; hybrid+BM25 = **stretch** | **hybrid in Phase 0** (Kimchi 1.1 — UC#1 is "Hybrid+rerank"). |
| Embeddings | **Nebius embeddings** default | **local `all-MiniLM`** default; Nebius = generation (the mandatory call). |
| Confidence grader | LLM **self-reports** `confidence` in the generate call | **separate grader** (JSON-mode / cosine fallback) — self-confidence is the weak design (Kimchi 1.2). |
| Schedule | chat UI Day 2 → **eval Day 3-4** | **eval green by Day 2 before UI polish** (the anti-pattern gate #5 guards). |

**AGY corrections we ABSORBED (folded into `design.md`):**
- **Corpus facts** — AGY's count was right and ours was wrong. Verified: **14 files = 7 PDF + 6 DOCX +
  1 PPTX**, guidebook **19.3 MB**. `design.md` §1 corrected; **PPTX excluded-and-logged in P0**,
  `PptxLoader` → Phase 1.
- **Exact-match / keyword eval question** — added to §7; it's the item that *measures* why hybrid beats
  dense-only (turns the P0 hybrid decision into a reported number).
- **Citation-grounding faithfulness check** — AGY listed it independently, corroborating the deterministic
  fallback added in §7. No change; confirmation that two independent passes treat LLM-judge as optional.

**Trivia still open (pick at build):** chunk overlap 50 (AGY) vs 64 (ours); recursive-character (AGY) vs
fixed-size (ours) chunker — both behind `Chunker`; faithfulness target 85% (AGY) vs 90% (ours), settle
after the spike; cloud vector store Qdrant (AGY, native sparse) vs Pinecone (ours, free credits).

**Useful as-is:** the AGY implementation plan's **structure** (`app/core/interfaces.py` ABCs, `eval/`,
`routes/`, `db/models.py`) matches our pure-core/thin-view + pluggable-interface design and is a good
scaffold to feed `writing-plans` — *after* the reconciliation above.

**Meta-learning:** an agent planning off a stale decisions doc faithfully reproduces the stale decisions.
Point every planning agent at `design.md` (the reviewed spec), not just `architecture-decisions.md`.

---

*Next: run the spike (`design.md` §9) → `writing-plans` for the phase-by-phase implementation plan.*
