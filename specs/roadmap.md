# Roadmap

*Canonical. Phases + **MUST vs SHOULD** + risk caps. Read before expanding scope. Status: review
incorporated (Kimchi, 2026-06-07), plan-ready pending the spike. A **demoable skeleton exists at the
end of every phase.***

## Guiding rule

**Phase 0 is finished — bot works end-to-end *and* the eval report exists with a scores table — before
any Phase 1 work begins.** Finishing the graded spine is the priority. **Scope creep is the #1 risk**;
the most likely way to fail is an impressive demo with a thin eval report.

**Hard timing rule:** the eval must be **runnable and produce a scores table by end of Day 2.** UI
polish (streaming, source cards) happens Day 3–4 *only if the eval is green.*

---

## Phase 0 — gradeable spine  *(MUST — this alone is a complete deliverable)*

**Goal:** member asks a course question → cited answer or honest refusal; the 15-question eval report
exists.

**MUST**
- Ingest the **commit-pinned GitHub eval corpus** (`awesome-agentic-ai-resources` +
  `Mastering-Agentic-AI-Week1`) via **`MarkdownLoader` + `JupyterLoader` + a GitHub fetcher**;
  **fixed-size chunking** (~512/64) with citation metadata (`repo, file_path, line_span` for GitHub;
  `doc_id, title, page/section, char_span` for files), behind a `Chunker` interface. **Never ingest the
  `Mastering-Agentic-AI-Week2` notebooks/code (the sample solution).**
- **Local `sentence-transformers` embeddings**; vectors in **Chroma**; metadata in **SQLite**.
- **Hybrid retrieval: dense + BM25 fused via RRF, `top_k=5`** (matches Use Case #1's pattern).
- **LangGraph** `grade → {answer + citations | refuse}`; grader = Nebius JSON-mode LLM call (fallback:
  cosine-similarity threshold). **Generation via Nebius** (the mandatory call).
- **Non-streaming chat UI** (form-post, HTMX) with `<details>` source cards.
- **2 seeded users** (admin, member) + session login.
- **Eval report:** 15 questions anchored to the **commit-pinned GitHub eval corpus** (≥6 edge cases incl.
  ~2 ambiguous + ~2 multi-document + ~3 unanswerable + ~2 chunking-stress — the handout's three named hard
  cases all covered; questions drafted in `../docs/design-review.md` Part 7). **Handout-mandatory spine =
  deterministic retrieval eval** (recall@k / precision@k / MRR) + failure-taxonomy table — *this* is
  "retrieval quality scores + where it fails and why." **LLM-as-judge faithfulness** (pinned rubric,
  temp 0, raw outputs saved) is a **depth add-on, cuttable** under rate-limit pressure → falls back to a
  deterministic **citation-grounding check**. Scores table carries retrieval columns always; faithfulness
  % from whichever scorer survives.
- **Gold-standard annotation (~6 h)** — start **Day 1**, parallel to scaffolding.

**SHOULD** — production-corpus file loaders (PDF/DOCX) + a **minimal admin upload endpoint** (the user
will add docs via UI). Eval ships on the pinned GitHub corpus regardless; polished upload UI = Phase 1.

**Risk cap:** if Nebius/parse surprises threaten the week, fall back to a smaller corpus subset and the
similarity-threshold grader. **Eval ships regardless.**

**Cut order if slipping:** SSE streaming → expandable source cards → admin upload UI → **LLM-judge
faithfulness (→ citation-grounding fallback)** → **(never)** the deterministic retrieval eval /
refusal path. The retrieval eval is the handout-graded artifact; the LLM-judge is not, so it is the
last thing cut before the protected core but it *can* be cut.

**Demoable skeleton:** ask → cited answer / refusal, plus the eval report with a scores table.

---

## Phase 1 — product layer  *(SHOULD — on a green core)*

**MUST (of this phase)**
- Real **admin vs member RBAC** + **invite-code** signup.
- **Admin content management:** upload, list, delete, re-index.
- **`usage_log` + usage dashboard:** queries over time, top questions, refusal rate, latency.
  *(Moved from Phase 0 — ungraded, so it doesn't tax the spine.)*

**SHOULD** — **web-page**, **PPTX**, Python/JSON ingestion added to the loader registry as the production
corpus grows; production tracks repo HEAD with a re-index trigger.

**Risk cap:** RBAC stays session-based; no OAuth/heavyweight provider unless genuinely cheap.

**Demoable skeleton:** admin logs in, uploads a doc, watches it become queryable, sees usage stats;
member logs in and chats.

---

## Phase 2 — depth & deploy  *(SHOULD / stretch — strong demo moments)*

- **Pinecone** preset — second `VectorStore` impl; live "Chroma → Pinecone, one config line".
- **Cross-encoder rerank** + **section-aware chunking** — each produces a before/after eval delta.
- **Nebius embeddings** preset — the "swap embedding provider" demo.
- **Deploy:** Docker → Hugging Face Space; Postgres preset; auth hardening; smoke-check live URL.

**Risk cap:** ship none of Phase 2 rather than ship Phase 0/1 unfinished. Each item independently
droppable.

---

## Cross-phase deliverables (handout)

- [ ] Working hybrid cited Q&A bot with refusal — end of Phase 0.
- [ ] 15-question eval report (recall/precision/MRR + faithfulness + taxonomy) — end of Phase 0.
- [ ] Demo video ≤5 min · GitHub repo · project write-up (incl. "divergences from sample solution").

## Resolved decisions

**Two-tier corpus** (eval = commit-pinned GitHub; production = repos@HEAD + uploads) · Week-2 repo excluded
(sample solution) · NotebookLM not integrated · auth (seeded→invite-code) · embeddings (local ST;
Nebius=generation) · chunking (fixed-size→section-aware P2) · hybrid in P0 · top-k=5 · synchronous ingest ·
19.3 MB PDF parse-gate+OCR-fallback (production). Details: `../docs/design.md` §8.
