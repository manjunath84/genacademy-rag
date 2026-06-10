# GenAcademy RAG Project Write-Up

## Overview

This project is a course-material RAG assistant for the GenAcademy Week 2 submission. It answers questions from a controlled corpus with citations and refuses when the available materials do not support an answer.

The deployed app is live at:

<https://Manjunath84-genacademy-rag.hf.space>

The repository is:

<https://github.com/manjunath84/genacademy-rag>

## What The App Demonstrates

- Hybrid retrieval with dense embeddings plus BM25, fused with reciprocal rank fusion.
- Line-level citations for retrieved course materials.
- A refusal path for unsupported questions.
- Nebius generation through an OpenAI-compatible provider seam.
- A deterministic retrieval eval with recall@k, precision@k, MRR, and failure analysis.
- A small product layer: login, invite-code signup, admin document management, and usage analytics.
- Docker deployment to Hugging Face Spaces with a live HTTP smoke test.

## Dataset And Corpus

The deterministic eval corpus is commit-pinned and intentionally small:

- `awesome-agentic-ai-resources`
- `Mastering-Agentic-AI-Week1`

The eval set contains 15 questions across:

- answerable questions
- exact-match questions
- chunking-stress questions
- multi-document questions
- ambiguous questions
- unanswerable questions

The Week 2 sample-solution repository is deliberately excluded and firewalled from ingest. This avoids copying or depending on the reference solution.

Production/demo uploads are separate from the eval corpus. Uploaded documents can be used in the live app, but they do not change the deterministic eval baseline.

## Architecture

The project keeps the core logic separate from the web and deploy layers:

- `src/genacademy_rag/core/`: chunking, retrieval, vector-store interface, provider interface, graph/pipeline behavior.
- `src/genacademy_rag/data/`: SQLite datastore for users, documents, invites, chunks, and usage logs.
- `src/genacademy_rag/web/`: FastAPI/Jinja/HTMX views and session handling.
- `src/genacademy_rag/deploy/`: Hugging Face Space bootstrap for first-boot eval corpus seeding.

The main extension seams are:

- provider preset: OpenRouter, OpenAI, Nebius, local Gemma-compatible endpoint
- vector store: Chroma by default, Pinecone preset available
- embeddings: local `sentence-transformers` by default, Nebius embeddings preset available
- chunker: fixed baseline plus section-aware chunking work

## Architecture Diagrams

The system diagrams are checked in under `docs/diagrams/`:

- [System overview](diagrams/01-system-overview.svg)
- [Query flow](diagrams/02-query-flow.svg)
- [Ingest and two-tier corpus](diagrams/03-ingest-two-tier-corpus.svg)
- [Hugging Face Space deployment](diagrams/04-deployment-hf-space.svg)

The editable Draw.io source is [architecture.drawio](diagrams/architecture.drawio).

## Prompts Used While Building

All prompts in this section are **reconstructed** — they are faithful paraphrases of what was asked based on the documented build history (git log, plan files, memory, and review artifacts). I do not have access to the raw transcript to quote verbatim. Each is labeled with its category and what it produced.

---

### 1. Planning / Architecture

**Prompt (reconstructed):**
> I'm building a course-material RAG assistant for the GenAcademy Week 2 submission. Requirements: answers with line-level citations, hard refusal path when out-of-corpus, hybrid retrieval (dense + BM25 + RRF), Nebius as the mandatory model call for generation, and a 15-question deterministic eval that must be green before I add any product or deploy features. The eval corpus is two-tier: a frozen commit-pinned snapshot for graded eval, and a growing production corpus for the live app. Help me design the architecture and write a phase breakdown.

**Produced:** `docs/design.md` — a full design with the two-tier corpus model, pluggable provider/retriever/chunker seams, pure core / thin view constraint, and Phase 0/1/2 phase definitions. The design established that "eval spine first" was the invariant, not optional.

---

**Prompt (reconstructed):**
> Before writing an implementation plan, I need to de-risk three unknowns: (1) GitHub fetch + commit pinning for the eval corpus, (2) parsing a 19.3 MB PDF guidebook without OCR, and (3) Nebius JSON mode for the refusal grader. Run the spike and report findings I can bake into the plan.

**Produced:** `docs/spike-findings.md` — confirmed commit SHAs for both eval repos, pypdf parsing was clean (printable ratio 0.994, no OCR needed), `all-MiniLM-L6-v2` dim=384, JSON mode working on open Llama via OpenRouter (~3.75 s cold, 10/10 no throttling). All three risks cleared, removing blockers before any code was written.

---

**Prompt (reconstructed):**
> I have a reviewed design at `docs/design.md` and spike findings at `docs/spike-findings.md`. Write a Phase 0 implementation plan with 17 TDD tasks ordered bottom-up: scaffold → types → provider → chunker → loaders → vectorstore → retriever → grader → LangGraph graph → datastore → ingest → query → gold set → retrieval eval → faithfulness → report → web → PDF/upload. Every RAG + data task must be unit-testable offline against a `FakeModelProvider` seam. Use `docs/architecture-decisions.md` for reasoning only — do not implement from it.

**Produced:** `docs/superpowers/plans/2026-06-08-genacademy-rag-phase0.md` — a 17-task implementation plan with explicit non-negotiables, spike facts baked in, and Phase 1–2 as a forward pointer only. This plan was then reviewed by two independent agents (Antigravity + Kimchi) before any code was written.

---

### 2. Implementation

**Prompt (reconstructed):**
> The Phase 0 plan has been reviewed by two independent agents and all findings are folded in. Execute it using subagent-driven development. Run tasks bottom-up, run `ruff` and `pytest` after each task, and gate each task on green before moving to the next. Do not start Phase 1 until the eval report is green.

**Produced:** Phase 0 implementation — 50 tests, ruff clean, eval `recall@k=0.67 / precision@k=0.22 / MRR=0.55 / faithfulness=58%`. Mandatory Nebius call satisfied. PR #1 merged.

---

**Prompt (reconstructed):**
> The Phase 0 PR review found a real refusal-bypass bug: `bool("false") == True` in Python, so a JSON-mode model that returns the string `"false"` was being treated as a passing answer. Add a regression test that sends the string `"false"` through the grader and asserts it refuses, then fix the parse.

**Produced:** 8 regression tests added, the `bool("false")` coercion fixed with strict string comparison, and the grader fallback path explicitly covered. Total tests rose from 50 to 58.

---

### 3. Code Review

**Prompt (reconstructed):**
> Review the Phase 0 PR diff for correctness bugs. I want line-level findings with file and line number evidence from the actual codebase — not general opinions. If a finding does not hold up against the code, drop it. Separate blocking issues from non-blocking ones.

**Produced:** Multi-agent PR review (4 agents) caught the `bool("false")` refusal-bypass, 9 additional issues including missing CSRF on destructive POSTs, and test gaps. Each finding included a file:line reference. Non-findings (e.g., a reviewer who applied the wrong project's visual-theme rules) were rejected with evidence.

---

**Prompt (reconstructed):**
> Review the Phase 1 design plan for security correctness — specifically invite-code hashing and corpus mutation consistency. I'm using bcrypt. Challenge the assumption that storing a bcrypt hash is enough to look up an invite code at redemption time. Also check whether a snapshot-swap is enough to protect concurrent dense + sparse retrieval, or whether two consistency domains require a lock around the entire retrieve call.

**Produced:** Two Codex review passes, both NEEDS-REWORK. Key findings accepted: invite codes need a structured `id.secret` format (bcrypt isn't lookupable — you need a clear id for O(1) lookup plus a secret half that's hashed); corpus threading.Lock must wrap the full `retrieve()` call not just mutations, because Chroma and BM25 are two consistency domains; `BEGIN IMMEDIATE` + conditional consume for atomic invite redemption. These were hardening decisions the builder would have missed.

---

### 4. Deployment

**Prompt (reconstructed):**
> Write a Docker + Hugging Face Space deployment plan for this FastAPI app. Requirements: first-boot eval corpus seeding via `deploy/bootstrap.py`, `GENACADEMY_DATA_DIR` for persistent storage, `GENACADEMY_SECURE_COOKIES` for Space environments, and a live HTTP smoke test that hits the login endpoint and validates a 200 response. The Space uses ephemeral `/data` unless paid storage is attached — document this limitation clearly.

**Produced:** `docs/superpowers/plans/2026-06-10-genacademy-rag-phase2-docker-hf-space-deploy.md`, then `deploy/bootstrap.py`, `Dockerfile`, `scripts/smoke_test.sh`, and `docs/deploy.md` runbook. Plan reviewed by Kimchi before code; PR #9 reviewed by 4 agents after. Post-merge hardening commit added.

---

### 5. Final Packaging

**Prompt (reconstructed):**
> Generate a project write-up for the Week 2 handout submission. Include: what the app demonstrates, the two-tier corpus design, architecture overview with seam descriptions, iterations tried (Phase 0 → 1 → 2 depth → deploy → live validation), key learnings, and divergences from the sample solution. Also include a section on prompts used while building — the handout explicitly asks for this.

**Produced:** `docs/project-writeup.md` (this file), `docs/demo-script.md`, and the `README.md` submission summary. The demo script covers the golden path (login → ask a course question → see citation → ask an unsupported question → see refusal) for the ≤5-minute cohort video.

---

**Key pattern across all prompts:** the most useful prompts required measurable outputs — eval deltas, specific file:line findings, test counts, or a gate before the next phase. Prompts that asked for "general feedback" or "your opinion" produced less actionable results than prompts that asked reviewers to verify a specific claim against the code.

## Iterations Tried

1. **Phase 0: gradeable spine**
   Built the basic cited Q&A flow, refusal behavior, commit-pinned corpus ingest, and retrieval eval.

2. **Phase 1: product layer**
   Added seeded users, invite-code signup, admin document management, upload/delete/reindex, and usage analytics.

3. **Phase 2 depth**
   Added reranking and section-aware chunking as measured retrieval-quality experiments while keeping deterministic eval defaults stable.

4. **Deploy slice**
   Added Docker packaging, Hugging Face Space startup, first-boot corpus seeding, secure-cookie settings, and HTTP smoke checks.

5. **Live validation**
   Deployed to Hugging Face Spaces, passed the live smoke check, verified a cited answer, and verified refusal on an unsupported question.

## Learnings

- RAG quality depends more on corpus boundaries, chunking, retrieval, and citations than on the model alone.
- Refusal behavior needs to be designed and tested as a first-class path, not treated as a fallback message.
- Deterministic evals are valuable because they make retrieval changes measurable instead of anecdotal.
- Deployment surfaces different risks than local tests: persistent storage, environment variables, startup bootstrap, and provider credentials all need explicit runbooks.
- Independent review was especially helpful for finding brittle assumptions around Docker, Hugging Face Spaces, and partial bootstrap state.

## Divergences From The Sample Solution

This project intentionally diverges from the handout sample solution:

- The Week 2 sample-solution repository was not fetched, read, or ingested.
- The eval corpus is commit-pinned and separate from production uploads.
- Retrieval quality is measured with a 15-question gold set and explicit recall/precision/MRR metrics.
- The app uses a LangGraph-style refusal branch rather than a generic always-answer chain.
- Hybrid retrieval and citation metadata are implemented directly in the project core.
- The web layer is a thin FastAPI/Jinja/HTMX shell over injected core services.
- The deploy path is Docker on Hugging Face Spaces with a live smoke script.

## Current Limitations

- Hugging Face `/data` is ephemeral unless paid persistent storage is attached, so users/uploads/usage reset on restart.
- The first deployment uses Chroma, not Pinecone, to keep the live demo simple.
- Rerank is disabled in the Space because the rerank model is not baked into the Docker image.
- The live HTTP smoke proves boot and login-page rendering; browser testing is still needed for actual query behavior.

## Final Submission Notes

Use these links in the cohort form:

- Live app: <https://Manjunath84-genacademy-rag.hf.space>
- GitHub repo: <https://github.com/manjunath84/genacademy-rag>

Suggested short description:

> GenAcademy RAG is a course-material assistant that answers with line-level citations and refuses unsupported questions. It uses hybrid retrieval, a pinned eval corpus, Nebius generation, admin document management, and a Docker Hugging Face Space deployment. The project is scale-ready by seams and evaluation discipline, not overbuilt with premature distributed infrastructure.
