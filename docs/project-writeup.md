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

The build used agent-assisted prompts for planning, implementation, and review. The most important prompt categories were:

- **Architecture planning:** asked for a phase-based RAG plan that protected the graded spine before adding product or deploy features.
- **Independent reviews:** asked separate agents to review the Docker/Hugging Face deploy plan and challenge assumptions against the actual codebase.
- **Code review follow-up:** asked reviewers to identify blocking issues, then fixed only the findings that held up against the code.
- **Deployment support:** asked for step-by-step Hugging Face Space setup, variable/secrets guidance, and live smoke validation.
- **Final packaging:** asked for a demo script, deployment instructions, and a concise project write-up.

The most useful review prompts were the ones that required line-level evidence from the codebase instead of general opinions.

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
