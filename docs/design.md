# GenAcademy RAG — Design

*Week-2 project (Gen Academy, "Grounding AI with RAG & Context Engineering"). Self-contained design
doc. Deep reasoning behind the locked stack: [`architecture-decisions.md`](architecture-decisions.md).
Independent review folded in: [`design-review.md`](design-review.md) (Kimchi, 2026-06-07).*

**Date:** 2026-06-07 · **Status:** review incorporated → **plan-ready pending the §9 spike.**

---

## 1. One-liner (the handout's required primer)

> *My RAG app helps **Gen Academy cohort members** answer **"what did the course say about X"
> questions** from **the cohort's curated materials — a *growing*, admin-owned corpus (Gen Academy
> GitHub repos + uploaded PDF/DOCX/PPTX files), not a fixed file set** — in a **web chat UI** with
> **≥90% faithfulness** and a **hard refusal path** when the answer isn't in the corpus.*

**Two-tier corpus (the organizing principle).** The curated material keeps changing, so the corpus splits:

| | **Eval corpus** (graded) | **Production corpus** (serves users) |
|---|---|---|
| Contents | Frozen, **commit-pinned** snapshot of the cohort's GitHub repos | The repos at HEAD **+** admin-uploaded files **+** future sources |
| Gold set | **ONE** 15-question gold set anchors here (§7) | none — never expands the gold set |
| Drift | Pinned → eval reproducible forever | Tracks HEAD; "re-index" on admin trigger |

This honors "the curated material keeps growing" without letting new content destabilize the graded
15-question spine (the #1 risk is under-budgeted gold annotation — one frozen gold set protects it).

- **Phase-0 eval corpus (commit-pinned):** the **`awesome-agentic-ai-resources`** repo (a curriculum
  catalog of 60+ resources in Markdown tables) **+** the **`Mastering-Agentic-AI-Week1`** hands-on repo.
  Loaded via `MarkdownLoader` + `JupyterLoader` over a **GitHub fetcher** pinned to a fixed commit SHA.
- **`Mastering-Agentic-AI-Week2` is EXCLUDED** — it **is** the handout's sample solution; ingesting or
  reading its notebooks/code is disqualifying (§2). An admin-uploaded Week-2 **PPT** may join the
  *production* corpus later; the repo's notebooks/code never do.
- **NotebookLM is not a source.** The shared notebook is a *sink* the user filled — no consumer API to
  pull from it. Its curated-resource *list* is exactly the `awesome-agentic-ai-resources` catalog we
  already ingest, so nothing extra is needed.
- **Production files (`../CuratedRAGMaterials/` + uploads):** the PDF/DOCX/PPTX exports (Week-1/2 decks,
  handouts, glossary, the *Mastering Agentic AI* guidebook PDF, **19.3 MB**) feed the **production**
  corpus via the file loaders + admin upload — additional retrievable content, **not** part of the graded
  gold set. The 19.3 MB guidebook is therefore a *production* parse-quality concern (§9/§10), **no longer
  a single point of failure for the graded eval** (which is clean GitHub-Markdown).
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

**Sample-solution firewall:** the `Mastering-Agentic-AI-Week2` repo **is** the handout's reference
solution (`1_rag_pipeline.ipynb`, `3_hybrid_rag.ipynb`, `company_kb_viewer.py`). Its notebooks/code are
**never** read or ingested — reading them to inform the build is disqualifying. Only opaque, non-code
admin uploads (e.g. a Week-2 PPT) may enter the *production* corpus later. The deliberate divergences are
documented in the write-up (§11) so a reviewer can't mistake this for a replicated starter.

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
| Relational DB | Pluggable `Datastore`: **SQLite** now; **Postgres** when persistence or multi-instance needs justify it. |
| Deployment | **Hybrid**: local-first, deploy-ready (Docker → HF Space). Deploy = Phase 2. |

**Pluggability rule:** interface + **one** implementation at MVP; the second impl is a Phase-2 demo.

## 5. Phasing (MVP-first)

### Phase 0 — gradeable spine (build *and finish* before anything else)
```
ingest pinned GitHub repos (Markdown+Jupyter) → chunk (fixed-size +citation metadata) → embed (local ST)
  → Chroma + BM25 → hybrid retrieve (RRF, k=5) → LangGraph[grade → answer+citations | refuse]
  → non-streaming chat UI (form-post) → EVAL REPORT
       (production corpus adds PDF/DOCX files + admin uploads on top — same pipeline, ungraded)
```
- **Roles, minimal:** one seeded **admin** + one seeded **member**, session login. (Eval corpus loaded by
  a commit-pinned script; a **minimal admin upload endpoint** is a Phase-0/1-boundary SHOULD — the user
  will add production docs via UI — while a polished upload UI is Phase 1.)
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
3. Admin upload **UI polish** → keep only a minimal upload endpoint (eval corpus is the commit-pinned
   script, so the graded spine never needs upload).
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
- **Deploy** (Docker → HF Space) + auth hardening; smoke-check live URL. Postgres remains a future
  preset for persistence or multi-instance needs, not a deploy prerequisite.

## 6. Architecture & data flow

**Pure core / thin view.** All logic in a testable core with **no** FastAPI/HTMX imports; the view is
the only HTTP/template layer.

### Interfaces (the pluggable seams)
- `Loader` registry — **`GitHubFetcher` + `MarkdownLoader` + `JupyterLoader` (Phase 0 — the
  commit-pinned eval corpus)**; `PdfLoader`, `DocxLoader` (production files, Phase 0 if cheap → Phase 1);
  `PptxLoader`, `PythonLoader`, `JSONLoader`, `WebLoader` (later, as the production corpus grows). *(One
  registry — eval vs production is which loaders run, plus a commit-pin on the eval set. `JSONLoader` was
  deferred because its only Phase-0 candidate, the ShopEasy KB, lives in the excluded Week-2 repo (§7).
  Adding a loader = a new class + config entry, never a refactor.)*
- `Chunker` — `FixedSizeChunker` (Phase 0) → `SectionAwareChunker` (Phase 2). *(Added per review: if
  chunking is an eval variable, it lives behind an interface.)*
- `ModelProvider` — `embed()` (local ST, Phase 0) + `generate()` (Nebius, mandatory).
- `VectorStore` — `ChromaStore` (Phase 0) → `PineconeStore` (Phase 2).
- `Retriever` — `HybridRetriever` (dense + BM25 + RRF, Phase 0) → + cross-encoder rerank (Phase 2).
- `Datastore` — users, documents, chunk metadata (+ usage log in Phase 1). *Watch scope:* split into
  `UserStore` / `DocStore` / `UsageStore` when Postgres or multi-instance persistence arrives; keep
  as one datastore until that pressure is real.

### Two pipelines
- **Ingestion (admin/script, offline):** `Loader → clean → Chunker(+metadata) → embed →
  VectorStore.upsert + BM25 index` + write `documents` / `chunks_meta` rows. *(Eval corpus pinned to a
  commit SHA per repo; production tracks HEAD and re-indexes on admin trigger.)*
- **Query (member, online):** `embed(query) → HybridRetriever.retrieve(k=5) → LangGraph[grade →
  answer | refuse] → {answer, citations}`. (Usage logging added in Phase 1.)

### Data model
- `users(id, email, role['admin'|'member'], created_at)`
- `documents(id, title, source_type['github'|'pdf'|'docx'|'pptx'|...], repo, file_path, commit_hash,
  filename, uploaded_by, status, n_chunks, created_at)` — `repo`/`commit_hash` set for GitHub sources;
  `filename` for uploads.
- `chunks_meta(id, doc_id, ordinal, page_or_section, line_start, line_end, char_start, char_end,
  text_preview)` — GitHub chunks carry line spans; file chunks carry page/char spans.
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

**The gold set is anchored to the commit-pinned eval corpus, and the eval runs only over it.** Kimchi
drafted 15 grounded questions (`design-review.md` Part 7) against `awesome-agentic-ai-resources/README.md`
+ Week-1 content; we adopt them with two reconciliations: **(a)** re-tag the three exact-value/acronym
items (Kimchi Q2/Q5/Q6) as our **exact-match** category below; **(b) Q8 (the ShopEasy KB) is
re-anchored** — its `shopeasy_knowledge_base.json` lives in the **excluded** Week-2 repo, so that
chunking-stress slot is re-filled by a split-table question over the `awesome-agentic-ai-resources`
Markdown tables (final wording locks during annotation). Each chunk records `commit_hash`; the retrieval
scorer checks it, so production content never leaks into the gold set.

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

**Schedule note:** gold-standard annotation for 15 questions over the **pinned eval repos** is **~6 h of
careful reading, not coding** — start Day 1, in parallel with scaffolding. (Markdown parses cleanly, so
annotation is lighter than over the 19.3 MB production PDF would have been.)

**Annotation gate — answer substance must be *in* the corpus, not just *catalogued*.** The
`awesome-agentic-ai-resources` repo is a **curriculum catalog** (links + short blurbs), so some of
Kimchi's "answerable" drafts may have no in-corpus substance — e.g. Q3 ("what does *Attention Is All You
Need* cover?") and Q6 ("what does QLoRA *do*?") are answerable only if the README **describes** them, not
merely **links** them. Before annotating, confirm each gold answer exists in the pinned corpus *text*; any
that don't either **move to the unanswerable bucket** (a correct refusal) or get **re-worded to what the
catalog actually states** ("which resource covers *Attention Is All You Need*?"). Likewise check Q9/Q10
are genuinely **multi-document** (e.g. README + a Week-1 notebook), not multi-*section* of the one README,
or the multi-doc retrieval claim is hollow. Re-balance to 4+2+2+2+2+3 after this pass.

## 8. Resolved decisions (were §8 open; closed via review)

| Decision | Resolution |
|---|---|
| **Corpus model** | **Two-tier:** eval = commit-pinned GitHub repos (one gold set); production = repos@HEAD + uploaded files + future sources. Honors a *growing* corpus without destabilizing the graded eval. |
| **Week-2 repo** | **Excluded** — it's the sample solution; notebooks/code never read or ingested. An admin-uploaded Week-2 PPT may join *production* later. |
| **NotebookLM** | **Not integrated** — a sink, no consumer API. Its resource list = the `awesome-agentic-ai-resources` catalog we already ingest. |
| **Auth model** | Seeded admin+member (Phase 0); **invite-code** signup (Phase 1). OAuth deferred — external dep not worth it in 4 days. |
| **Embeddings** | **Local `all-MiniLM-L6-v2` (384-dim)** Phase 0; Nebius = **generation** (the mandatory call); Nebius embeddings = Phase 2 swap demo. |
| **Chunking** | **Fixed-size + overlap (~512/64)** Phase 0 behind `Chunker`; section-aware = Phase 2 comparison. |
| **Retrieval depth** | **Hybrid dense+BM25+RRF** in Phase 0 (matches Use Case #1's pattern, ~30-min add); rerank Phase 2. |
| **top-k** | **k=5** Phase 0 (eval sensitivity-tests it). |
| **Ingestion** | **Synchronous with progress** (HTMX); no background worker at this corpus size. |
| **19.3 MB guidebook** | **Production** file (not eval-gating). **Parse-quality gate + per-doc chunk cap** (see §9); OCR fallback; exclude-if-bad and note it. |

## 9. Pre-build spike (do first; ~45 min — now gates more)

Against the **live Nebius/Pinecone endpoints and the real corpus**:
- Nebius **chat model ID**, and whether it supports **JSON mode / structured output** (decides the
  grader + judge format).
- **Throughput / rate limits:** fire ~10 sequential requests; check for free-tier throttling (the eval
  runs 15 Q × generate + 15 judge calls in a loop).
- Per-call **latency** (embed-local + generate) → confirm the < 8 s ceiling holds with the chosen grader.
- **GitHub fetch + commit-pin (eval corpus):** list/fetch raw files at a fixed SHA; Markdown + Jupyter
  parse cleanly; record the SHA the gold set anchors to.
- **Parse-quality gate on the 19.3 MB guidebook (production, not eval-blocking):** char density not mostly
  whitespace/garbled, expected headings present, mean chunk length above a floor; if it fails → OCR
  (`pdf2image`+`pytesseract` / `marker`); if still bad → exclude and log.
- Pinecone free-tier credits + index config (dimension must match 384).

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Production parse quality** (slide decks extract badly; 19.3 MB PDF may be image-heavy). **No longer the #1 *eval* risk** — the graded eval is clean GitHub-Markdown — but still gates production retrieval. | Parse-quality gate + OCR fallback + exclude-if-bad (§9); guidebook is production, not eval-blocking. |
| **Corpus mutability** (GitHub repos change; line numbers/wording drift) | Eval pinned to a commit SHA; `commit_hash` in chunk metadata; scorer checks it. Production tracks HEAD with a re-index trigger. |
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
      (eval-first split, pluggable interfaces, LangGraph refusal graph, hybrid retrieval, the two-tier
      eval/production corpus, NotebookLM-independent ingestion) so a reviewer can't mistake it for a
      replicated starter.

---

*Next: run the §9 spike → `writing-plans` for the phase-by-phase implementation plan.*
