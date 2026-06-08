# GenAcademy RAG — Design

*Week-2 project (Gen Academy, "Grounding AI with RAG & Context Engineering"). Self-contained design
doc. Deep reasoning behind the locked stack: [`architecture-decisions.md`](architecture-decisions.md).
Independent review folded in: [`design-review.md`](design-review.md) (Kimchi, 2026-06-07).*

**Date:** 2026-06-07 · **Status:** review incorporated → **plan-ready pending the §9 spike.**

---

## 1. One-liner (the handout's required primer)

> *My RAG app helps **Gen Academy cohort members** answer **"what did the course say about X"
> questions** from **the cohort's curated materials (14 files — PDF + DOCX + one PPTX, owned by the Gen
> Academy team)** in a **web chat UI** with **≥90% faithfulness** and a **hard refusal path** when the
> answer isn't in the corpus.*

- **Corpus (verified 2026-06-07 against `../CuratedRAGMaterials/`):** **14 files = 7 PDF + 6 DOCX + 1
  PPTX**, ~73 MB total. Week-1/2 session decks (PDF), project handouts (DOCX), the Week-1 glossary (PDF,
  1.6 MB), the *Mastering Agentic AI* guidebook (PDF, **19.3 MB** — the parse-quality risk in §9/§10),
  token-usage best-practices (DOCX), session chat-question docs (DOCX), and one "Read Later" deck
  (`Week 1 — Read Later (Deck 3).pptx`, 3.7 MB). Admin-owned; refreshes per cohort week.
- **PPTX handling (Phase 0 decision):** the single PPTX is **excluded from the Phase-0 corpus and logged
  as a known exclusion** in the eval report; a `PptxLoader` (`python-pptx`) is a Phase-1 loader-registry
  add, not P0 scope. Rationale: one "Read Later" deck doesn't justify a third loader on the graded spine;
  excluding-and-logging is itself the handout's "document where it fails / what's out of scope" discipline.
- **Faithfulness, not just relevance:** answer must be grounded in retrieved chunks; measured in §7.
- **Latency ceiling:** < 8 s end-to-end. The grader mechanism (§7) is chosen partly to protect this.

Maps to handout **Use Case #1 (Enterprise Policy Q&A Bot)** — retrieval pattern **"Hybrid + rerank"** —
plus the bonus **Chatbot UI** add-on, **Track 2 (code: LangChain + LangGraph)**, **Nebius** for the
mandatory model call.

## 2. What this is — and what it is NOT

**Is:** a multi-user internal knowledge product over **non-sensitive** course materials. Admins manage
the corpus + monitor usage; members chat with cited answers.

**Is NOT** an extension of `legal-rag-private`. That is a **separate** portfolio piece with a
privacy/on-prem/regulated-docs thesis (local Gemma). This is a **fresh standalone build (Branch B)**,
own repo, cloud generation (Nebius), cloud-deployable. The privacy thesis **does not apply here and
must not be claimed.** The only carry-over is a *transferable eval lesson* (§7), not code.

## 3. What is actually graded (the anchor for all phasing)

Use Case #1 deliverables: **(1)** a working **cited** Q&A bot; **(2)** a **15-question evaluation
report** with retrieval-quality scores + **failure analysis**, including **ambiguous**,
**multi-document**, and **unanswerable** cases; **(3)** demo video (≤5 min) + GitHub + write-up.
Handout thesis, verbatim: *"Most RAG projects do not fail at the model. They fail at chunking,
retrieval quality, or evaluation."* → **eval is a first-class deliverable.**

## 4. Locked stack (summary; reasoning in `architecture-decisions.md`)

| Layer | Choice |
|---|---|
| Service / UI | **FastAPI + HTMX + Tailwind + Alpine.js** — one service, server-session auth, one Docker image. **Phase 0 UI is non-streaming form-post**; SSE streaming is an optional enhancement (first to cut). |
| RAG orchestration | **LangChain** primitives + **one** small **LangGraph** graph (refusal/escalation branch only). **Pin** `langchain*` / `langgraph` to exact versions. |
| Generation | **Nebius Token Factory** — the mandatory cohort call. |
| Embeddings | **Local `sentence-transformers` `all-MiniLM-L6-v2` (384-dim)** for Phase 0 (free, offline, deterministic). Nebius embeddings = Phase 2 swap demo. |
| Chunking | **Fixed-size + overlap (≈512 / 64 tok)** Phase 0, behind a **`Chunker`** interface. Section-aware = Phase 2 eval axis. |
| Retrieval | **Hybrid: dense (Chroma) + BM25 (`rank-bm25`) fused via RRF**, `top_k=5`, in **Phase 0**. Cross-encoder **rerank** = Phase 2. |
| Vector store | Pluggable `VectorStore`: **Chroma** (Phase 0) → **Pinecone** (Phase 2). |
| Relational DB | Pluggable `Datastore`: **SQLite** (Phase 0) → **Postgres** (deploy). |
| Deployment | **Hybrid**: local-first, deploy-ready (Docker → HF Space). Deploy = Phase 2. |

**Pluggability rule:** interface + **one** implementation at MVP; the second impl is a Phase-2 demo.

## 5. Phasing (MVP-first)

### Phase 0 — gradeable spine (build *and finish* before anything else)
```
ingest PDF+DOCX → chunk (fixed-size +citation metadata) → embed (local ST) → Chroma + BM25
  → hybrid retrieve (RRF, k=5) → LangGraph[grade → answer+citations | refuse]
  → non-streaming chat UI (form-post) → EVAL REPORT
```
- **Roles, minimal:** one seeded **admin** + one seeded **member**, session login. (Corpus seeded by a
  script; admin upload UI is a cut-candidate, not required here.)
- **Citations:** every chunk stamped at ingest `{doc_id, title, page/section, char_span}`; carried to
  the answer; rendered as `<details>` source cards.
- **Refusal path (graded):** LangGraph `retrieve → grade → {answer | "I could not find this in the
  course materials"}`. **Grader = LLM call via Nebius JSON mode** returning
  `{"answerable": bool, "confidence": 1-5}` (pending §9 spike); **fallback = max cosine-similarity
  threshold** (calibrated on 3–5 held-out questions) if JSON mode is absent or blows the latency budget.
- **Eval report (deliverable):** §7. Gold-standard annotation (~6 h) starts **Day 1** in parallel.

**Definition of done:** member asks → cited answer or honest refusal; the 15-question eval report
exists with a **scores table** (recall@k, precision@k, MRR, faithfulness, refusal correctness) + a
**failure-analysis table**. *This alone is a complete, gradeable submission.*

**Cut order if the schedule slips (never cut eval or refusal):**
1. SSE streaming → plain form-post (already the baseline).
2. Expandable source cards → footnote links.
3. Admin upload UI → seed corpus via script.
4. LLM-judge faithfulness (falling back to citation-grounding check).
5. **Never:** the 15-question eval report or the refusal path.

### Phase 1 — product layer (the vision, once Phase 0 is green)
- Real **admin vs member RBAC** + signup/gating (**invite-code**, see §8).
- **Admin content management:** upload, list, delete, re-index.
- **`usage_log` + usage dashboard:** queries over time, top questions, refusal rate, latency.
  (Moved here from Phase 0 — ungraded, so it doesn't tax the spine.)
- **Web-page ingestion** added to the loader registry.

### Phase 2 — depth & deploy (stretch; strong demo moments)
- **Pinecone** preset — second `VectorStore` impl; live "Chroma → Pinecone, one config line".
- **Cross-encoder rerank** + **section-aware chunking** — each a before/after eval delta.
- **Nebius embeddings** preset — the "swap the embedding provider" demo.
- **Deploy** (Docker → HF Space) + Postgres preset + auth hardening; smoke-check live URL.

## 6. Architecture & data flow

**Pure core / thin view.** All logic in a testable core with **no** FastAPI/HTMX imports; the view is
the only HTTP/template layer.

### Interfaces (the pluggable seams)
- `Loader` registry — `PdfLoader`, `DocxLoader` (Phase 0) → `PptxLoader`, `WebLoader` (Phase 1). *(The
  corpus's 1 PPTX is excluded-and-logged in P0 per §1; the registry shape means adding it later is a new
  class + config entry, not a refactor.)*
- `Chunker` — `FixedSizeChunker` (Phase 0) → `SectionAwareChunker` (Phase 2). *(Added per review: if
  chunking is an eval variable, it lives behind an interface.)*
- `ModelProvider` — `embed()` (local ST, Phase 0) + `generate()` (Nebius, mandatory).
- `VectorStore` — `ChromaStore` (Phase 0) → `PineconeStore` (Phase 2).
- `Retriever` — `HybridRetriever` (dense + BM25 + RRF, Phase 0) → + cross-encoder rerank (Phase 2).
- `Datastore` — users, documents, chunk metadata (+ usage log in Phase 1). *Watch scope:* split into
  `UserStore` / `DocStore` / `UsageStore` when the Postgres preset arrives; keep as one for Phase 0.

### Two pipelines
- **Ingestion (admin/script, offline):** `Loader → clean → Chunker(+metadata) → embed →
  VectorStore.upsert + BM25 index` + write `documents` / `chunks_meta` rows.
- **Query (member, online):** `embed(query) → HybridRetriever.retrieve(k=5) → LangGraph[grade →
  answer | refuse] → {answer, citations}`. (Usage logging added in Phase 1.)

### Data model
- `users(id, email, role['admin'|'member'], created_at)`
- `documents(id, title, filename, source_type, uploaded_by, status, n_chunks, created_at)`
- `chunks_meta(id, doc_id, ordinal, page_or_section, char_start, char_end, text_preview)`
- `usage_log(...)` — **Phase 1.**

## 7. Eval plan (first-class deliverable)

**Two evals, deliberately ranked — the handout grades the first; the second is a depth signal we add.**
The handout's deliverable is *"a 15-question evaluation report with **retrieval quality scores** and
**failure analysis** … document where retrieval succeeds, where it fails, and why."* That is a
**retrieval** evaluation — no LLM required. The faithfulness judge is above-and-beyond (our own
≥90%-faithfulness claim + the `honest-ai-app` thesis), not a handout requirement, so it is **cuttable**
under rate-limit pressure where the retrieval eval is not.

1. **Retrieval eval — deterministic, NO LLM. THE HANDOUT-MANDATORY SPINE; never cut.** Per question, a
   gold marker / gold `doc_id`+section. Report **recall@k, precision@k, and MRR** (review: recall alone
   is thin). Catches chunking/retrieval bugs without LLM nondeterminism. Used for the **dense-only vs
   dense+BM25** before/after. This alone satisfies "retrieval quality scores + where it fails and why."
2. **End-to-end faithfulness eval — LLM-as-judge, pinned. DEPTH SIGNAL; on the cut list.** A
   **verbatim** judge prompt taking `{question, answer, retrieved_chunks}` →
   `{"faithful": bool, "hallucinated_claims": [..], "score": 1-5}`, run at **temperature 0**; **raw
   judge outputs saved in the repo** for auditability. (Nebius JSON mode if the spike confirms it; else
   a rigid regex-parsed format.) Costs ~30 extra Nebius calls (15 generate + 15 judge) in a loop — a
   real exposure on a rate-limited free tier (§9). **If the spike shows throttling, cut this and fall
   back** to the deterministic check below; the graded retrieval eval ships regardless.
   - **Deterministic faithfulness fallback (zero LLM):** a **citation-grounding check** — do the
     answer's cited spans actually appear in the retrieved chunks? Partial faithfulness signal (catches
     fabricated citations, not paraphrase drift), but keeps a faithfulness column in the report with no
     Nebius cost. This is what runs if the LLM-judge is cut.

**The 15-question set** (≥6 edge cases; covers the handout's required hard cases):
- 4 **answerable straightforward** · 2 **exact-match / keyword** (a rare proper noun or exact phrase —
  e.g. a specific tool or term that appears verbatim in one doc; **BM25 should win where dense misses**) ·
  2 **chunking-stress** (answer spans a chunk boundary / a caption separated from its context) · 2
  **multi-document** · 2 **ambiguous** · **3 unanswerable** (one about something the corpus never
  mentions, one related-but-not-covered, one adversarially close to corpus terms but absent). *(Review:
  1 unanswerable is not a stress test. The exact-match item is the eval question that **measures why
  hybrid beats dense-only** — it turns the Phase-0 hybrid decision into a reported number. Ranges sum to exactly 15; exact distribution locked during question construction.)*

**Failure taxonomy (pre-defined so the table is scorable):** `ChunkingBoundary`,
`RetrievalRecallFailure`, `FaithfulnessHallucination`, `RefusalFalsePositive`, `RefusalFalseNegative`,
`TopKTooSmall`.

**Outputs:** a scores table (recall@k · precision@k · MRR · refusal correctness, **always**; faithfulness %
from the LLM-judge **or** the citation-grounding fallback) + a **failure-analysis table**
(Symptom → Cause[taxonomy] → Fix). The retrieval columns + failure table *are* the handout's required
deliverable; faithfulness % is the depth add-on.

**Schedule note:** gold-standard annotation for 15 questions over ~20 files is **~6 h of careful
reading, not coding** — start Day 1, in parallel with scaffolding.

## 8. Resolved decisions (were §8 open; closed via review)

| Decision | Resolution |
|---|---|
| **Auth model** | Seeded admin+member (Phase 0); **invite-code** signup (Phase 1). OAuth deferred — external dep not worth it in 4 days. |
| **Embeddings** | **Local `all-MiniLM-L6-v2` (384-dim)** Phase 0; Nebius = **generation** (the mandatory call); Nebius embeddings = Phase 2 swap demo. |
| **Chunking** | **Fixed-size + overlap (~512/64)** Phase 0 behind `Chunker`; section-aware = Phase 2 comparison. |
| **Retrieval depth** | **Hybrid dense+BM25+RRF** in Phase 0 (matches Use Case #1's pattern, ~30-min add); rerank Phase 2. |
| **top-k** | **k=5** Phase 0 (eval sensitivity-tests it). |
| **Ingestion** | **Synchronous with progress** (HTMX); no background worker for ~20–25 files. |
| **20 MB guidebook** | **Parse-quality gate + per-doc chunk cap** (see §9); OCR fallback; exclude-if-bad and note in eval. |

## 9. Pre-build spike (do first; ~45 min — now gates more)

Against the **live Nebius/Pinecone endpoints and the real corpus**:
- Nebius **chat model ID**, and whether it supports **JSON mode / structured output** (decides the
  grader + judge format).
- **Throughput / rate limits:** fire ~10 sequential requests; check for free-tier throttling (the eval
  runs 15 Q × generate + 15 judge calls in a loop).
- Per-call **latency** (embed-local + generate) → confirm the < 8 s ceiling holds with the chosen grader.
- **Parse-quality gate on the 20 MB guidebook:** char density not mostly whitespace/garbled, expected
  headings present, mean chunk length above a floor; if it fails → OCR (`pdf2image`+`pytesseract` /
  `marker`); if still bad → exclude and log.
- Pinecone free-tier credits + index config (dimension must match 384).

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Corpus parse quality** (slide decks extract badly; 20 MB PDF may be image-heavy) — #1 schedule risk | Parse-quality gate + OCR fallback + exclude-if-bad (§9). |
| **Gold-standard annotation underestimated** (~6 h, manual) | Scheduled explicitly, started Day 1 in parallel. |
| **"Impressive demo, weak eval"** — days spent on UI/streaming, eval thrown together Day 4 | **Hard rule:** eval runnable + scores table **by end of Day 2**; UI polish only if eval green. Cut order (§5). |
| **Nebius free-tier rate limits** | Throughput check in the spike; batch/sleep in the eval loop if throttled. |
| **LangChain/LangGraph version churn** | **Pin exact versions** in `pyproject.toml` + lockfile. |
| Scope creep past a finished Phase 0 | Roadmap MUST/SHOULD + cut order; `usage_log`/dashboard deferred to Phase 1. |
| Refusal grader fires constantly / never | Calibrate threshold (or LLM grader) on held-out questions; ≥3 unanswerable Qs validate it. |

## 11. Deliverables checklist (handout)

- [ ] Working **hybrid-retrieval** cited Q&A bot with refusal path (Phase 0).
- [ ] 15-question evaluation report (recall/precision/MRR + faithfulness + failure taxonomy).
- [ ] Demo video ≤5 min (walkthrough + how AI tools were used + live result).
- [ ] GitHub repo (own repo, separate from legal-rag).
- [ ] Project write-up (Google Doc) — incl. a **"Design divergences from the sample solution"** section
      (eval-first split, pluggable interfaces, LangGraph refusal graph, hybrid retrieval) so a reviewer
      can't mistake it for a replicated starter.

---

*Next: run the §9 spike → `writing-plans` for the phase-by-phase implementation plan.*
