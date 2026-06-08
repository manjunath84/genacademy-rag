# Mission

*Canonical. Read before changing scope. Status: draft, pending external design review.*

## Why this exists

Gen Academy cohort members constantly ask "what did the course actually say about X?" — and the answer
is buried across a dozen lecture decks, handouts, a glossary, and guidebooks. This is a **grounded,
cited Q&A assistant over the cohort's curated materials** so a member gets a trustworthy, sourced
answer in seconds — or an honest "that isn't in the materials" instead of a confident hallucination.

It is also a **portfolio artifact**: a multi-user internal knowledge *product* (RBAC, admin content
management, usage monitoring) that demonstrates RAG depth distinct from the `legal-rag-private`
regulated-docs piece.

## Audience

- **Cohort members (primary users):** technical builders (often Java/Spring), newer to AI. They chat;
  they want fast, cited, plain answers they can trust and verify against the source.
- **Admins (Gen Academy team):** upload/manage the corpus, monitor usage. Want a simple, reliable
  content + analytics surface.

## Success looks like

- A member asks a course question and gets a **faithful, cited** answer (or a correct refusal) in
  < ~8 s.
- The graded deliverables ship: working bot + **15-question eval report** (retrieval scores +
  faithfulness + failure analysis) + demo + repo + write-up.
- The architecture visibly supports **swapping** data sources, model providers, and retrieval
  strategies (interface + config, demonstrated by a second implementation).

## In scope

- Multi-format ingestion (Phase 0: PDF, DOCX; Phase 1: web pages), chunking with citation metadata.
- Hybrid retrieval (dense + BM25, Phase 0) → cited generation → refusal path. Cross-encoder rerank = Phase 2.
- Two roles (Phase 0: seeded admin + member; Phase 1: real RBAC + signup).
- Admin content management + usage dashboard (Phase 1).
- Local-first, deploy-ready (Phase 2: Docker → HF Space).
- A first-class **evaluation** harness + report.

## Out of scope (explicitly)

- The `legal-rag-private` **privacy / on-prem thesis** — false here (cloud API, non-sensitive docs).
- Real-time collaboration, multi-tenant orgs, billing.
- Replicating the handout's sample solutions or the `Knowledge-Intelligence-System` reference repo.
- Anything that delays a **finished Phase 0** (the graded spine + eval).

## The one-liner (handout primer)

> My RAG app helps **Gen Academy cohort members** answer **"what did the course say about X" questions**
> from **the cohort's curated materials (~15–25 PDF/DOCX files, owned by the Gen Academy team)** in a
> **web chat UI** with **≥90% faithfulness** and a **hard refusal path** when the answer isn't in the
> corpus.
