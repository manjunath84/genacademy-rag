# Design Review — GenAcademy RAG

*Reviewer posture: independent, skeptical, pre-implementation. This review is written to surface disagreements, not validate assumptions. It assumes the handout, the design doc, and the architecture-decisions log as read simultaneously.*

---

## TL;DR

Phase 0 is **directionally correct** but is missing two load-bearing pieces for the graded submission: **hybrid retrieval** and **robust gold-standard construction time**. The eval plan is conceptually sound (retrieval-recall separate from faithfulness) but underspecified on scorer automation and edge-case coverage. Architecture is clean but the confidence-grader mechanism is a blank spot. The biggest schedule risk is not coding — it is **corpus parsing + gold-standard annotation**. The design is **not ready to turn into an implementation plan** until three blocking items are resolved (hybrid scope, grader mechanism, PDF-parse fallback). Two other items (eval automation, unanswerable coverage) are strong should-fix.

---

## 1. BLOCKING — Must resolve before implementation planning

### 1.1 Use Case #1 explicitly says "Hybrid + rerank"; dense-only Phase 0 is a grading liability

**Concern:** The handout lists Use Case #1 (Enterprise Policy Q&A Bot) with retrieval pattern **"Hybrid + rerank"**. The design deems dense-only retrieval "already gradeable" and defers hybrid + BM25 + rerank to Phase 2. This is a high-variance bet. The grader selected this use case specifically; the framework text says "Hybrid retrieval is usually right" and "Pure dense misses exact matches (error codes, names, ticker symbols)." A Phase 0 submission that does not demonstrate hybrid retrieval may score lower on "retrieval quality" — the exact thesis the handout says most projects fail on.

**Why it matters:** You are not being graded on what you *plan* to add in Phase 2. You are graded on the submitted bot and eval report. If the evaluator looks at Use Case #1's prescribed pattern and does not see it in the deliverable, the justification "we planned it for later" does not help.

**Concrete recommendation:** Add a **minimal hybrid retriever to Phase 0** — dense (Chroma) + a lightweight BM25 layer over the same chunk text, merged with a simple reciprocal rank fusion or even a weighted sum. `rank-bm25` is 30 minutes of integration. It does not need a cross-encoder rerank yet. The eval report then compares dense-only vs. dense+BM25 on the 15-question set. This single addition turns the eval from a description of what is missing into a before/after measurement, which the handout explicitly rewards.

*Quoting design §5:* "Dense retrieve → LangGraph[grade → answer | refuse]". The Phase 0 pipeline should read: "**Hybrid** retrieve → ..."

---

### 1.2 The LangGraph confidence-grader is completely underspecified

**Concern:** The design says Phase 0 includes "LangGraph: retrieve → grade confidence → {answer + citations | refuse}" but never specifies **how** confidence is graded. Is it a cosine-similarity threshold on the top retrieved chunk? A separate LLM call asking "is this question answerable from the retrieved context?"? An entropy score? A relevance classifier? Each option has very different latency, cost, and correctness properties — and the design cannot be built without picking one.

**Why it matters:** The refusal path is a **graded deliverable** (the unanswerable eval question). If the grader is an ad-hoc threshold that fires constantly or never fires, the eval fails. If it is a second LLM call, the 8-second latency ceiling becomes 2-3 LLM round-trips (embed → retrieve → grade → generate), which is hard to hit. If it is a similarity threshold, the threshold itself is a hyperparameter that needs calibration.

**Concrete recommendation:** Specify the grader **before** locking the plan. Two viable options:

- **Option A (LLM grader):** An inexpensive, fast model call (qwen/lightweight via Nebius) with a structured prompt returning `{"answerable": bool, "confidence": 1-5}`. Budgets ~500 ms. Requires JSON mode support — validated in the §9 spike.
- **Option B (heuristic grader):** Compute the max cosine similarity between query embedding and top-k retrieved chunks. If max similarity < threshold (e.g., 0.72 for all-MiniLM, calibrated on 3-5 held-out questions), refuse. Zero extra latency. Simpler. Less robust.

Pick one in the plan. Do not leave it as "grade confidence" in the architecture diagram.

---

### 1.3 The 20 MB guidebook PDF is a single point of failure with no fallback

**Concern:** The design says "may need per-doc chunk caps" and "verify parse in spike" but provides **no fallback** if the guidebook parses poorly. A 20 MB PDF is likely image-heavy or scanned. If pdfplumber / PyPDF produces garbage text, the largest document in the corpus silently poisons retrieval. The design does not say "exclude it if parsing fails" or "have a second parser ready."

**Why it matters:** If the guidebook generates 500 junk chunks, every query will retrieve some percentage of junk, dragging faithfulness down. The eval will fail, and there will be no time to fix it mid-week.

**Concrete recommendation:** In the §9 spike, add a **parse-quality gate**: after loading, check that the extracted text has reasonable character density (not 90% whitespace / garbled unicode), has expected section headings, and produces chunks with mean length above a floor. If it fails, switch to a second parser (e.g., `pdf2image` + `pytesseract` OCR, or `marker`). If that also fails, **exclude the guidebook from the corpus** and note it in the eval report. Do not let one file sink the week.

---

## 2. SHOULD-FIX — Resolve before or during build; design ships without these but is weaker

### 2.1 Only 1 unanswerable question in the 15-question set is too few

**Concern:** The design's proposed split is ~9 straightforward, ~3 multi-doc, ~2 ambiguous, ~1 unanswerable. The unanswerable case is the **only** question that grades the refusal path. With exactly one question, there is **zero statistical confidence** — if the question happens to trigger a false positive (the model hallucinates an answer using priors, or the retriever accidentally pulls a weakly-related chunk), the refusal path looks correct when it is not, or looks broken when it is fine.

**Why it matters:** "Stress-test" implies repetition, not a single probe. A 1/15 sample is not a stress test.

**Concrete recommendation:** Increase to **at least 3 unanswerable questions** (different domains: one about something the corpus never mentions, one about a related-but-not-covered topic, one that is adversarially close to corpus terms but actually absent). This is the difference between a checkbox and a genuinely useful failure analysis.

---

### 2.2 The eval's faithfulness scorer is "LLM-as-judge rubric and/or manual scoring" — pick one

**Concern:** The eval plan says faithfulness is "judged via an LLM-as-judge rubric and/or manual scoring." "And/or" is a specification gap. For a reproducible, gradeable report, the scoring method must be deterministic and documented. Manual scoring is fine but needs a rubric and a second rater for inter-rater reliability. LLM-as-judge is faster but needs the exact judge prompt, the judge model, and temperature=0. Mixing the two without a rule for when to use which makes the scores incomparable.

**Why it matters:** The eval report is a graded deliverable. "We sort of eyeballed it" or "we used an LLM sometimes" will not read as rigorous.

**Concrete recommendation:** Commit to **LLM-as-judge with a pinned rubric** for Phase 0. Define a structured prompt (copied verbatim into the spec) that takes `{question, answer, retrieved_chunks}` and returns `{"faithful": bool, "hallucinated_claims": [str], "score": 1-5}`. Run it with temperature=0. If Nebius JSON mode works (§9 spike), this is clean structured output. Save the raw judge outputs in the repo so the report is auditable. If Nebius does not support JSON mode, use regex parsing with a very rigid output format.

---

### 2.3 The chunking open decision should close on fixed-size for Phase 0, with section-aware as the Phase 2 eval axis

**Concern:** The design wants section/heading-aware chunking because "decks have slide/section structure." This is true in theory and false in practice for many PDF slide decks. Slide titles are often images, not parseable text. Section boundaries in DOCX are more reliable but still require a heading parser. Building a section-aware chunker for PDF is non-trivial and a time sink.

**Why it matters:** The handout explicitly says Use Case #2 rewards comparing fixed vs semantic chunking. This project is Use Case #1, so the comparison is not required. But the design's ambiguity on chunking means the builder may lose half a day building a fragile heading parser.

**Concrete recommendation:** Close the open decision: **fixed-size chunking with overlap for Phase 0** (predictable, fast, easy to tune). Document chunk size and overlap in the spec (e.g., 512 tokens, 64-token overlap). Make **section-aware chunking** a Phase 2 eval axis: run the same 15 questions with fixed vs. section-aware, measure recall delta, and include the comparison in the eval report. This turns a build risk into a demonstration of technical depth.

---

### 2.4 The first thing to cut if the schedule slips is the streaming UI, not the eval

**Concern:** The design's Phase 0 includes "Minimal chat UI (HTMX + SSE) with expandable source cards." SSE streaming is harder than it looks, especially with cited answers that need to render expandable cards mid-stream. The design correctly says the eval is a named deliverable, but does not explicitly state which Phase 0 item is the **first to sacrifice** if Day 3 arrives and eval is not done.

**Why it matters:** Scope creep is identified as the #1 risk, but the escape hatch is not labeled.

**Concrete recommendation:** Declare the **cut order** in the plan:
1. **First cut:** Streaming / SSE. Drop to a simple form-post UI that returns the full answer + citations on page reload. HTMX swap still works; streaming is a nice-to-have.
2. **Second cut:** Expandable source cards. Render citations as plain `<details>` blocks or footnote links.
3. **Third cut:** The admin upload UI. Seed the corpus via a one-time script instead.
4. **Never cut:** The 15-question eval report or the refusal path.

---

## 3. OPTIONAL — Would strengthen the design but are not blockers

### 3.1 Missing risk: gold-standard annotation time is underestimated

The design does not list "creating gold standards for 15 questions" as a schedule risk. For a corpus of ~20 files including a 20 MB guidebook, manually identifying the correct chunk(s) for each question takes 15-30 minutes per question = **4-8 hours of work**. This is not a coding task; it is a careful reading task. It must be in the schedule.

**Recommendation:** Add "Gold-standard annotation: ~6 hours" to the Phase 0 schedule explicitly. Start it on Day 1 in parallel with scaffold setup.

### 3.2 Missing risk: evaluator may compare against the handout's sample solution

The handout says: "We highly encourage you NOT to look at this before... Use them as a hint document rather than replicating the following solutions. If you end up replicating the following solutions, you will not be given scores." The AGENTS.md correctly warns against this. However, the design does not discuss **how to deliberately diverge** from the sample solution in a visible way. A reviewer who sees a generic LangChain RAG chain may assume replication.

**Recommendation:** In the project write-up (Google Doc), include a "Design divergences from the sample solution" section that explicitly calls out: (a) the eval-first approach (retrieval-recall separate from faithfulness), (b) the pluggable interface design, (c) the LangGraph refusal graph, and (d) the hybrid retrieval plan. Make the divergence intentional and documented.

### 3.3 The usage_log table is in Phase 0 but serves no graded purpose

The design includes `usage_log` in Phase 0 because "the data already exists" for Phase 1. But writing usage_log rows adds database code, schema, and testing time to Phase 0 for a feature that is not graded. It is a small cost, but in a 4-day build, small costs add up.

**Recommendation:** Move `usage_log` to Phase 1. If truly trivial, keep it but do not test it beyond a smoke test.

### 3.4 The Nebius spike should test rate limits and cold-start latency, not just capability

The §9 spike tests model availability, JSON mode, and rough latency. It does not test **throughput** — what happens when you run 15 eval questions + 15 judge calls + embedding calls in a tight loop? Free-tier APIs often have aggressive rate limits.

**Recommendation:** Add a rate-limit check to the spike: fire 10 sequential requests and measure if any are throttled.

---

## 4. Detailed evaluation of the 7 required areas

### 4.1 The MVP-first phasing (design §5)

Phase 0 contains the right bones: ingest → chunk → embed → retrieve → generate → refuse → minimal UI → eval. The boundary between Phase 0 (gradeable spine) and Phase 1 (product layer) is conceptually clean. However, two items are **misplaced**:

- **Hybrid retrieval belongs in Phase 0, not Phase 2.** As argued in §1.1, Use Case #1's prescribed pattern is "Hybrid + rerank". Deferring the hybrid part to Phase 2 means the graded submission does not match the use case's declared retrieval pattern. The BM25 layer is cheap enough to include now.
- **The eval report construction belongs in Phase 0, but the schedule does not allocate enough time for it.** Building the eval harness (code) is one thing. Creating 15 good questions with gold standards and running failure analysis is 1-2 days of work. It is listed as a Phase 0 deliverable but not sized.

Phase 1 items (RBAC, admin dashboard) are correctly deferred. Phase 2 items (Pinecone, deploy) are correctly stretch.

**Verdict:** Phase 0 is 80% correct. Move hybrid retrieval in, add 6 hours for gold-standard annotation, and it is a solid gradeable spine.

---

### 4.2 The eval plan (design §7)

The **two-eval split** (deterministic retrieval-recall vs. LLM faithfulness) is the strongest part of the design. It is the transferable lesson from `legal-rag-private` and correctly prevents the "conflated signal" problem. This is genuinely good architecture for evaluation.

**Weaknesses:**

- **Edge-case coverage is thin.** Only 1 unanswerable question (§2.1). Only 2 ambiguous. The handout says "stress-test with 15 questions including edge cases." A 6/15 edge-case ratio would be more credible: 8 straightforward, 2 multi-doc, 3 ambiguous, 2 unanswerable.
- **No mention of chunking-specific eval questions.** The handout thesis is "RAG projects fail at chunking, retrieval quality, or evaluation." The eval should include questions that are **known to break with bad chunking** — e.g., a question whose answer spans a chunk boundary, or a question about a figure caption that gets separated from its image description. Without these, the eval checks faithfulness but not chunking quality.
- **Failure analysis format is under-specified.** "Symptom → Cause → Fix" is good, but the taxonomy of causes matters. The design should pre-define categories: ChunkingBoundary, RetrievalRecallFailure, FaithfulnessHallucination, RefusalFalsePositive, RefusalFalseNegative, TopKTooSmall, etc. This makes the table scorable and comparable.
- **Precision is absent.** Recall@k is mentioned, but not precision@k or mean reciprocal rank (MRR). For a 15-question eval, report at least **recall@k, precision@k, and MRR**.

**Verdict:** Conceptually sound, executionally thin. Boost edge-case count, add chunking-stress questions, define the failure taxonomy, and report precision + MRR.

---

### 4.3 Scope realism

**The schedule will slip on corpus ingestion and gold-standard annotation, not on code.**

Coding the RAG pipeline with LangChain is a solved problem — maybe 4-6 hours for a competent builder. The interfaces add maybe 2 hours of boilerplate. The UI is maybe 3-4 hours (non-streaming). The LangGraph graph is 1-2 hours once the grader mechanism is specified.

**The hidden time sinks:**

1. **PDF parsing:** ~2-4 hours. Slide decks are notoriously bad for text extraction. The 20 MB guidebook could eat a full day if OCR is needed.
2. **Gold-standard annotation:** ~6-8 hours. 15 questions across 20+ files, identifying the exact correct chunks, is slow, careful work.
3. **Eval debugging:** ~3-4 hours. Running the eval, finding failures, diagnosing whether it is chunking or retrieval, tweaking parameters, re-running.
4. **Nebus integration surprises:** ~1-2 hours. Any new API has quirks.

Total realistic build time: **3-4 days of focused work.** It is doable, but there is no slack. The minute PDF parsing goes sideways or the corpus has an unexpected format, Phase 0 scope must contract.

**What to cut first:** As specified in §2.4, streaming UI → expandable cards → usage_log → admin upload.

**Verdict:** Tight but doable with aggressive scope guarding. The design correctly identifies scope creep as the #1 risk, but does not put a hard boundary on Phase 0 scope.

---

### 4.4 Architecture soundness (design §6, tech-stack.md)

**Pure core / thin view:** Correct. The boundary between FastAPI/HTMX and the testable core is well-drawn. The binding guardrail in tech-stack.md ("No fastapi imports inside core/") is appropriate.

**Pluggable interfaces:** Good discipline. Each interface (`ModelProvider`, `VectorStore`, `Retriever`, `Datastore`) has one Phase 0 implementation and a planned Phase 2 second implementation. This satisfies the "swappable" requirement without over-engineering.

**One LangGraph graph only:** Correctly scoped. The design says linear steps stay LCEL; only the refusal branch uses LangGraph. This is the right balance — LangGraph is a depth signal, not load-bearing for correctness.

**Missing seam / premature abstraction concern:** The `Datastore` interface wraps users, documents, chunk metadata, and usage logs. This is four different concerns behind one interface. For an MVP, it is fine. But it risks becoming a God Interface that is hard to implement for Postgres later. Consider splitting into `UserStore` + `DocStore` + `UsageStore` when the Postgres preset is built. For Phase 0, keep it as one.

**Another gap:** There is no `Chunker` interface. The design talks about chunking strategy as an open decision (fixed vs section-aware) and even suggests comparing them as an eval axis, but there is no `Chunker` in the interface list. If chunking is a variable, it should be behind an interface too.

**Verdict:** Sound overall. Add a `Chunker` interface. Watch the `Datastore` scope creep.

---

### 4.5 Open decisions (design §8)

Taking positions on each:

| Decision | Position | Reasoning |
|---|---|---|
| **Auth model** | **Seeded users Phase 0; invite-code Phase 1.** | OAuth adds external dependency (Google Cloud console, callback URLs, secrets rotation) that is not worth 4 hours in a 4-day build. Invite-code is a simple DB table of codes. Seeded users prove the role concept with zero build time. |
| **Embeddings provider** | **Local `sentence-transformers` (all-MiniLM-L6-v2, 384-dim) for Phase 0; Nebius embeddings as Phase 2 swappable demo.** | The mandatory Nebius call should be **generation**, not embeddings. This is cleaner (one mandatory call, clearly identified). Local embeddings are free, deterministic, offline, and fast. They remove a network dependency from the ingest pipeline and the eval. Nebius embeddings can be added in Phase 2 as the "swap provider" demo. |
| **Chunking strategy** | **Fixed-size with overlap for Phase 0; section-aware as Phase 2 comparison.** | Section-aware chunking requires a reliable heading parser. PDF slide decks often have titles as images, not text. Fixed-size is predictable, easy to tune, and fast to implement. The Phase 2 eval can compare fixed vs section-aware to show technical depth. |
| **Ingestion processing** | **Synchronous with progress bar.** | Background workers (Celery, RQ, threads) add infrastructure complexity. For a corpus of 20-25 files, a synchronous ingest that streams progress via HTMX is fine. Admin uploads a file, waits 30 seconds, sees confirmation. |
| **20 MB guidebook PDF** | **Parse-quality gate + per-doc chunk cap.** | As specified in §1.3: gate on parse quality, fall back to OCR, exclude if still bad. Also apply a chunk cap: if the guidebook produces >N chunks (e.g., 50% of total corpus), downsample or reduce its chunk size to prevent retrieval skew. |

**Additional open decision not listed:** What is the **top-k** for retrieval? The framework asks for it. The design does not specify. Recommend k=5 for Phase 0. This is a tunable parameter that the eval can sensitivity-test.

---

### 4.6 Risks (design §10)

**What is missing from the risk list:**

1. **Corpus parsing quality (already discussed in §1.3).** This is the #1 unlisted risk.
2. **Gold-standard annotation time (§3.1).** The eval is not just code; it is careful manual work.
3. **Nebius rate limits / free-tier restrictions (§3.4).** The spike tests capability, not throughput.
4. **LangChain version churn.** The design does not pin LangChain/LangGraph versions. A mid-week release could break the pipeline. Pin `langchain`, `langchain-community`, `langchain-nebius`, `langgraph` to exact versions in `pyproject.toml`.
5. **The "impressive demo, poor grade" scenario (§3.2).** The design mentions "builder-of-the-week tempting more product surface" but does not describe the specific failure mode: a beautiful UI with streaming and dashboards, but a shallow eval report with vague failure analysis and no metrics. This is the most likely way to lose points.

**What is the most likely way this ends with an impressive demo but poor grade?**

The builder spends Days 1-3 on a polished HTMX chat UI with SSE streaming, admin upload, and usage_log tables. Day 4 morning, they panic-assemble 15 questions, run them once, eyeball the answers, and write a thin eval report. The report has no recall@k numbers, no faithfulness scores, no failure taxonomy — just a table saying "Question 7 failed, maybe chunking?" The demo video looks great. The GitHub repo is clean. But the graded deliverable (the eval report) is weak. **This is the specific failure mode to guard against.**

**Recommendation:** Add a hard rule to AGENTS.md: "The eval report must be runnable and produce a scores table by the end of Day 2. UI polish is Day 3-4 only if the eval is green."

---

### 4.7 Stack fit

**FastAPI + HTMX:** The right call. The architecture-decisions doc makes a solid case: one Python service, no JS build, server-session auth, one Docker image. For a 4-day build, this is faster than React + Vite + separate frontend. The concern is SSE streaming, but as noted in §2.4, that is the first thing to cut. The design correctly identifies HTMX as the complexity-minimizing choice.

**One valid alternative worth mentioning:** **Streamlit.** It would be even faster (no HTMX fragments, no Jinja templates, built-in chat widget). But Streamlit's session management is weaker for multi-user auth, and the portfolio signal is lower. FastAPI + HTMX is defensible.

**Nebius:** Sound for the mandatory call. The design correctly uses the OpenAI-compatible SDK with a `base_url` swap. The only concern is relying on Nebius for both embeddings and generation in Phase 0, which the design leaves ambiguous (§8). My recommendation: use Nebius for generation only, local ST for embeddings.

**Pinecone:** Phase 2 only is correct. Chroma is sufficient for Phase 0 and avoids a network dependency during the critical path.

**LangChain + LangGraph:** Track 2 is correctly chosen. The handout says "default to Track 2 if you write code regularly." The builder does. The one-LangGraph-graph rule is appropriately scoped.

**Verdict:** Stack is coherent and correctly matched to the time constraint. The only change is to route only generation through Nebius in Phase 0.

---

## 5. Final verdict

This design reflects **good architectural taste** — the pure core/thin view, the pluggable interfaces, the eval-first separation of retrieval-recall and faithfulness, and the ruthless Phase 0 focus are all signs of a builder who has learned from prior RAG projects. The design correctly identifies the refusal path and the eval report as load-bearing.

**However, it is not yet ready to turn into an implementation plan** because three items are still blank spots that would force mid-build improvisation:

1. **How confidence is graded in the LangGraph refusal branch** (§1.2).
2. **Whether hybrid retrieval is in Phase 0** (§1.1) — the handout's Use Case #1 says it should be.
3. **What happens if the 20 MB guidebook does not parse cleanly** (§1.3).

Once these three are resolved — preferably with a 30-minute impl spike for the BM25 layer and a 15-minute parse test on the guidebook — the plan can proceed. The two strong should-fix items (eval automation method, unanswerable question count) should be decided during planning but will not block implementation.

If the builder follows the cut order in §2.4, guards against the "impressive demo, weak eval" failure mode, and pins LangChain versions, this is a **high-probability gradeable submission**.

---

## Part 6: Scope-Change Assessment (GitHub Corpus — June 7)

**Triggered by:** User revealed corpus is not `../CuratedRAGMaterials/` PDFs/DOCXs, but live GitHub repos under `The-Gen-Academy` org.

### 6.1 What I inspected

| Repo | Files | Content |
|---|---|---|
| `awesome-agentic-ai-resources` | `README.md` (~28K chars) | 6-week curriculum catalog with 60+ resources in markdown tables |
| `Mastering-Agentic-AI-Week1` | Jupyter notebooks, Python | Session hands-on exercises (LangChain Basics) |
| `Mastering-Agentic-AI-Week2` | `1_rag_pipeline.ipynb` (240KB), `2_metadata_filtering.ipynb`, `3_hybrid_rag.ipynb`, `company_kb_viewer.py` (Streamlit), `shopeasy_knowledge_base.json` | **This is the reference solution for Week 2** |
| Future repos | Unknown | From `The-Gen-Academy` org, unspecified |

### 6.2 Critical finding: Reference solution collision

The `Week2` repo **is the sample solution** the handout warns against replicating.

- `1_rag_pipeline.ipynb` builds: Pinecone + OpenAI `text-embedding-3-small` + GPT-4.1-mini + `RecursiveCharacterTextSplitter`
- `2_metadata_filtering.ipynb` adds: Pinecone metadata filters + LLM-classified structured output
- `3_hybrid_rag.ipynb` adds: LangChain `EnsembleRetriever` with RRF fusion
- `company_kb_viewer.py` is a Streamlit app with KB browser + chat agent

**Verdict:** Our divergence (HTMX/Tailwind, local `sentence-transformers`, Nebius host, LangGraph refusal graph, own RRF fusion, eval-first graveyard) is our only protection against scoring zero. **This must be explicitly documented in the README and demo video.**

### 6.3 Architecture deltas vs. the existing design

| Design doc assumption | Reality | Required change |
|---|---|---|
| `FileLoaderRegistry` (PDF, DOCX) | GitHub repos (Markdown, Jupyter, Python, JSON) | `GitHubLoaderRegistry` with `MarkdownLoader`, `JupyterLoader`, `PythonLoader`, `JSONLoader` |
| Citation = `(doc_id, page, section)` | Line-based + commit hash | Add `repo`, `file_path`, `line_start/end`, `commit_hash` |
| Static file system | Live repos with new commits | Pin eval to commit hash; production fetches HEAD; stale indicator |
| `CharacterTextSplitter` on clean text | Markdown tables split across rows | Table-aware chunking OR accept split-table chunks (they make great eval stress tests) |

### 6.4 Ingestion requirements for Phase 0

To load the actual corpus, the system needs:

1. **MarkdownLoader** — parse `README.md`, preserve heading hierarchy, handle tables
2. **JupyterLoader** — convert `.ipynb` to markdown via `nbformat`, extract code cells as metadata
3. **JSONLoader** — parse `shopeasy_knowledge_base.json`, one `Document` per entry with metadata
4. **GitHub fetcher** — list repo contents (single dir or full tree), fetch raw files, track commit SHA

**Out of scope for MVP:** Incremental/delta updates. Use snapshot-at-sync. Rebuild index on admin trigger.

### 6.5 Eval consequences: corpus mutability

The eval 15-question set is gold-annotated to specific lines/sections. GitHub commits change line numbers and wording.

**Fix:**
- Lock the eval corpus to a specific commit hash per repo
- Record `commit_hash` in each chunk's metadata
- The retrieval scorer checks `chunk.meta.commit_hash == EXPECTED_COMMIT`

**Production vs. eval split:**
| Mode | Commit | Behavior |
|---|---|---|
| `eval` | Pinned | Always loads same snapshot; eval reproducible forever |
| `production` | `HEAD` | Always latest; stale indicator in UI |

### 6.6 Recommendation on which repos to ingest

| Repo | Include? | Rationale |
|---|---|---|
| `awesome-agentic-ai-resources` | **Yes** | Primary content — curriculum catalog. Safe. |
| `Mastering-Agentic-AI-Week1` | **Yes** | Hands-on exercises. Safe. |
| `Mastering-Agentic-AI-Week2` | **Risky** | Is the reference solution. Ingesting it into your RAG (which answers user questions about course materials) is fine *functionally*. **Reading its code to inform your own implementation is not.** If you must include it, treat it as opaque data (do not open the notebooks while designing Phase 0). |
| Future repos | Admin-add | Admin pastes a URL; system fetches and appends. No auto-discovery of org repos at build time. |

---

## Part 7: Eval Question Draft (15 questions, grounded in inspected content)

All answers anchored to `awesome-agentic-ai-resources/README.md` unless noted.

### Straightforward (6)

| # | Question | Expected Answer Anchor | What it tests |
|---|---|---|---|
| 1 | "What is the learning objective for Week 2?" | "...context injection, chunking, and smart routing" | Heading→paragraph retrieval |
| 2 | "How long does the DeepLearning.AI and Weaviate RAG Short Course take?" | "4 hours" | Table cell, exact value |
| 3 | "What topics does 'Attention Is All You Need' cover?" | "Self-attention... Transformer architecture... multi-head attention" | Dense retrieval on paper title |
| 4 | "Which Week 5 optional dive covers building neural nets from scratch?" | "Neural Networks: Zero to Hero — Andrej Karpathy" | Optional section retrieval |
| 5 | "What benchmark compares embedding models?" | "MTEB Leaderboard" | Acronym/proper noun |
| 6 | "What does QLoRA stand for, and what does it do?" | "Quantized Low-Rank Adaptation... reduces memory usage during fine-tuning" | Acronym + definition |

### Chunking-stress (2)

| # | Question | Expected Answer Anchor | What it tests |
|---|---|---|---|
| 7 | "The RAG short course is recommended for week 2. What optional deeper dives also cover RAG?" | "How I Got Better at Evaluating LLMs", "AI Engineering — Chip Huyen", "LLM Powered Autonomous Agents" | Multi-row collection across optional section; answer spans three table rows |
| 8 | "What are the five document types in the ShopEasy knowledge base, and how many of each exist?" | `past_ticket` (18), `runbook` (10), `product_doc` (10), `faq` (7), `bug_report` (6) | From `Week2-Session1/readme.md`; table parsing + exact counts |

### Multi-document (2)

| # | Question | Expected Answer Anchor | What it tests |
|---|---|---|---|
| 9 | "Compare the learning objectives of Week 1 and Week 2. What changes?" | W1: "Foundational concepts... prompting strategies"; W2: "Build systems with context-awareness... context injection, chunking" | Cross-section synthesis |
| 10 | "Which resources mention embeddings across multiple weeks?" | W1: "Embeddings: What they are..."; W2: "RAG Short Course", "The Embedding Archives"; W5: "MTEB Leaderboard" | Broad retrieval, deduplication, week-level grouping |

### Ambiguous (2)

| # | Question | Valid Answers | Scoring rubric |
|---|---|---|---|
| 11 | "What should I read about chunking?" | Primary: "Chunking Strategies for RAG" (Weaviate); Secondary: RAG Short Course, Embedding Archives, LlamaIndex docs | Accept if primary is named; bonus if secondary mentioned. Penalty if it fabricates a resource not in the catalog. |
| 12 | "What tools are recommended for running local LLMs?" | llama.cpp, Ollama, vLLM (Week 5) | Accept 2+ tools. QLoRA is fine-tuning, not running — acceptably adjacent but not perfect. |

### Unanswerable (3)

| # | Question | Expected Behavior | What it tests |
|---|---|---|---|
| 13 | "What is the learning objective for Week 8?" | Refusal: "The knowledge base only covers Weeks 1–7" or similar | Out-of-bounds hallucination resistance |
| 14 | "What are the due dates for each week's assignments?" | Refusal | Boundary between curriculum resources and assignment schedules |
| 15 | "Who is the specific instructor for Week 4?" | Refusal: "Instructors are not named in this knowledge base" | Prevents hallucination of a name |

---

## Part 8: Document-Change Handling Strategy

1. **Eval stability:** Pin to a `COMMIT_HASH`. Eval only loads `git show <hash>:README.md`, never `HEAD`. This guarantees the 15 questions never drift.

2. **Production freshness:** Admin-triggered sync. After sync, `last_synced` timestamp shown in UI. If `HEAD` commit ≠ `last_synced_commit`, show: "New content available — click to re-index."

3. **New repos:** Admin pastes a URL in a simple form. No auto-discovery of the entire org (scope creep for a 4-day build).

4. **JSON eval schema:** Each eval entry should include:
   ```json
   {
     "question": "...",
     "gold_doc_ids": ["awesome-agentic-ai-resources/README.md"],
     "gold_commit": "abc1234...",
     "gold_answer": "...",
     "category": "straightforward|chunking_stress|multi_doc|ambiguous|unanswerable",
     "failure_mode_if_wrong": "retrieval_miss|hallucination|refusal_failure|fragmentation"
   }
   ```

---

**Signed review, completed:** Second-pass assessment of design changes + independent review of GitHub corpus scope change.

*Review updated: 2026-06-07*
