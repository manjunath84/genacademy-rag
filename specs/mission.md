# Mission

*Canonical. Read before changing scope. Status: draft, pending external design review.*

## Why this exists

Gen Academy cohort members constantly ask "what did the course actually say about X?" — and the answer
is buried across a dozen lecture decks, handouts, a glossary, and guidebooks. This is a **grounded,
cited Q&A assistant over the cohort's curated materials** so a member gets a trustworthy, sourced
answer in seconds — or an honest "that isn't in the materials" instead of a confident hallucination.

It is also a **portfolio artifact** and a reusable Gen Academy internal knowledge *product*: the same
app should serve this cohort first, then remain useful for future cohorts as admins add or refresh
cohort materials. RBAC, admin content management, and usage monitoring are product requirements, not
demo-only extras, and the architecture should stay extensible without becoming a speculative platform.

## Audience

- **Cohort members (primary users):** technical builders (often Java/Spring), newer to AI. They chat;
  they want fast, cited, plain answers they can trust and verify against the source.
- **Admins (Gen Academy team):** upload/manage this cohort's corpus, refresh materials for future
  cohorts, and monitor usage. Want a simple, reliable content + analytics surface.

## Success looks like

- A member asks a course question and gets a **faithful, cited** answer (or a correct refusal) in
  < ~8 s.
- The graded deliverables ship: working bot + **15-question eval report** (retrieval scores +
  faithfulness + failure analysis) + demo + repo + write-up.
- Gen Academy admins can reuse the app for future cohorts by extending or replacing the **production
  corpus** without rewriting the core RAG, auth, or admin architecture.
- The architecture visibly supports **swapping** data sources, model providers, and retrieval
  strategies (interface + config, demonstrated by a second implementation).

## Corpus model (the curated material *grows* — it is not a fixed file set)

The cohort's materials keep changing (new repos, admin uploads, future sources), so the corpus is **two-tier**:

- **Eval corpus (graded):** a frozen, **commit-pinned** snapshot of the cohort's GitHub repos. The one
  15-question gold set anchors here, so the graded eval is reproducible forever.
- **Production corpus (serves users):** the repos at HEAD **+** admin-uploaded files (PDF/DOCX/PPTX) **+**
  future sources. Grows freely; **never** expands the gold set.

This lets the material keep growing without destabilizing the graded spine (under-budgeted gold
annotation is the #1 risk — one frozen gold set protects it).

## In scope

- **Two-tier ingestion.** Phase 0 eval corpus = commit-pinned GitHub repos via Markdown/Jupyter loaders +
  a GitHub fetcher. Production = + admin-uploaded PDF/DOCX/PPTX files (and later web pages). Chunking with
  citation metadata throughout.
- Hybrid retrieval (dense + BM25, Phase 0) → cited generation → refusal path. Cross-encoder rerank = Phase 2.
- Two roles (Phase 0: seeded admin + member; Phase 1: real RBAC + signup).
- Admin content management + usage dashboard (Phase 1).
- Local-first, deploy-ready (Phase 2: Docker → HF Space).
- A first-class **evaluation** harness + report.

## Out of scope (explicitly)

- The `legal-rag-private` **privacy / on-prem thesis** — false here (cloud API, non-sensitive docs).
- Real-time collaboration, multi-tenant orgs, billing.
- Replicating the handout's sample solutions or the `Knowledge-Intelligence-System` reference repo.
- **Ingesting or reading the `Mastering-Agentic-AI-Week2` repo's notebooks/code** — it *is* the sample
  solution; reading it to inform the build is disqualifying. (Week-2 contributes nothing for now; an
  admin-uploaded Week-2 **PPT** may join the *production* corpus later — never the repo's code.)
- **Integrating with NotebookLM** — it's a *sink*, not a source (no consumer API). Its curated-resource
  *list* is already the `awesome-agentic-ai-resources` catalog we ingest directly.
- Anything that delays a **finished Phase 0** (the graded spine + eval).

## The one-liner (handout primer)

> My RAG app helps **Gen Academy cohort members** answer **"what did the course say about X" questions**
> from **the cohort's curated materials (a *growing*, admin-owned corpus — Gen Academy GitHub repos plus
> uploaded PDF/DOCX/PPTX files; not a fixed file set)** in a **web chat UI** with **≥90% faithfulness**
> and a **hard refusal path** when the answer isn't in the corpus.
