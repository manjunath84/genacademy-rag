# GenAcademy RAG Minimal System Design Posture

**Date:** 2026-06-10
**Status:** review draft, documentation only
**Audience:** GenAcademy Week 2 project reviewers and future maintainers

## Purpose

This document captures the minimal system-design ideas worth applying to the GenAcademy RAG project
without turning a Week 2 course submission into an overbuilt production platform.

The project context is intentionally modest:

- target users: GenAcademy cohort members
- cohort size: about 200 people
- cohort cadence: one cohort per month
- corpus: course materials that grow over time
- priority: cited answers, honest refusal, reproducible eval, and clear extension seams

The system should be easy to demonstrate today and credible to scale later. It should not add Week
3-6 agentic features, multi-agent orchestration, broad tool routing, or heavy infrastructure unless a
measured failure justifies them.

## Sources Reviewed

Local project docs:

- `docs/design.md`
- `specs/roadmap.md`
- `AGENTS.md`
- `docs/learnings.md`
- `docs/phase0-decisions-and-tradeoffs.md`
- `docs/phase1-decisions-and-tradeoffs.md`
- `docs/phase2-decisions-and-tradeoffs.md`

Parent-folder system-design notes:

- `../notes-agentic-rag-production-architecture.md`
- `../notes-building-production-rag-systems.md`

Local production-corpus files observed under `data/`:

- Week 1 and Week 2 decks and handouts in PDF, DOCX, and PPTX form
- `Mastering-Agentic-AI-Getting-Started-Guidebook.pdf`
- chat-question documents
- glossary and token-optimization notes
- local Chroma and SQLite state

The repository intentionally ignores `/data/`, so these local files are not committed artifacts.

## Current System-Design Baseline

The project already has several production-shaped choices:

- pure core / thin FastAPI view
- two-tier corpus model:
  - eval corpus is frozen and commit-pinned
  - production corpus can grow through uploads and local course files
- cited chunk provenance captured at ingest
- refusal path before unsupported generation
- hybrid retrieval with dense search plus BM25 and RRF
- optional cross-encoder rerank, measured before enabling
- configurable vector store and embedding provider presets
- SQLite datastore behind a datastore seam
- admin upload, delete, reindex, invite, and usage-log flows
- deterministic retrieval eval with a fixed gold set

This is enough architecture for the current cohort scale. The right next moves are documentation,
deploy readiness, and lightweight operating discipline, not larger agentic behavior.

## Scale Assumptions

For the next few cohorts, assume:

- hundreds of users, not tens of thousands
- low-to-moderate concurrency
- repeated questions around assignments, tools, readings, and session logistics
- monthly corpus growth as each cohort receives new materials
- correctness and trust matter more than shaving every millisecond

The system should optimize for:

- reproducible eval
- source-grounded answers
- fast local development
- clear demo story
- simple deployment
- future migration paths that are visible but not implemented prematurely

The system should not optimize yet for:

- high-QPS serving
- multi-region deployment
- distributed tracing
- sharded vector indexes
- Redis-backed semantic caches
- multi-agent query planning
- generalized web or SQL tool routing

## Minimal Architecture Recommendation

Keep the near-term architecture as:

```text
FastAPI app
  -> session auth and admin UI
  -> QueryPipeline
  -> HybridRetriever
  -> local embeddings or Nebius embeddings preset
  -> Chroma or Pinecone preset
  -> SQLite datastore
  -> Nebius/OpenAI-compatible generation provider
```

This is the correct shape for the project because it keeps each scale axis behind a seam:

- vector store: Chroma now, Pinecone when remote vector serving matters
- relational data: SQLite now, Postgres when multi-instance persistence matters
- embeddings: local now, Nebius when provider swap is the demo or remote ingest is needed
- retrieval quality: rerank and chunking experiments stay measured and optional
- deployment: Docker/Hugging Face Space without changing core logic

## Data Folder Treatment

The new files in `data/` should be treated as a production/demo corpus, not eval data.

Recommended policy:

- keep `/data/` gitignored
- never add these files to the deterministic eval corpus
- ingest them only through production upload or a production corpus bootstrap path
- keep the eval baseline pinned to GitHub commit SHAs
- document which files were used in the demo corpus if they appear in the video

Loader priority:

1. PDF support remains highest value because many course assets are PDFs.
2. DOCX support is useful for handouts and chat-question docs.
3. PPTX support can wait unless slides are central to the final demo.

The key invariant is that production-corpus growth must never mutate or re-score the fixed eval
collection.

## Minimal Production Concepts To Adopt Now

### 1. Operating Assumptions

Add a short project-doc section that states:

- expected user count and cohort cadence
- expected corpus growth pattern
- current deployment target
- acceptable latency posture
- what is intentionally not implemented

This makes the design look intentional instead of incomplete.

### 2. Observability With Existing Usage Logs

Use the existing `usage_log` and admin dashboard as the first observability layer.

Track and report:

- query count
- refusal rate
- latency
- fallback usage
- top repeated questions
- number of citations used

Do not add Prometheus, Grafana, OpenTelemetry, or centralized logs yet. Those are future production
upgrades after the app has real traffic.

### 3. Cost Discipline

Keep generation single-pass by default.

Near-term cost controls:

- keep `top_k=5` unless eval proves otherwise
- avoid LLM router calls
- avoid LLM reflection loops
- avoid web search fallback
- keep faithfulness judge optional
- use exact-query caching only if repeated cohort questions show up in usage logs

Cost should be measured before adding cost infrastructure.

### 4. Deployment Readiness

The active deploy plan is the right next slice:

- Docker packaging
- Hugging Face Space metadata
- first-boot corpus bootstrap
- secure cookie setting
- HTTP smoke script
- deploy runbook

This gives a credible live demo path without requiring Postgres, Redis, queues, or multi-worker
retriever state.

### 5. Evaluation As A Release Gate

Keep eval as the strongest system-design control.

Every retrieval-quality change should include:

- before/after metrics
- per-question movement
- failure interpretation
- default-on or default-off recommendation

This is already working well for rerank, section-aware chunking, Pinecone, and Nebius embeddings.

## Minimal Constitution Addendum Proposal

Do not edit `AGENTS.md` yet. Reviewers can decide whether to adopt this language.

Proposed addendum:

```markdown
## System Design Posture

This project is scale-aware, not scale-overbuilt. Design for about 200 learners per monthly cohort
unless a new requirement changes that assumption.

- Add infrastructure only when it protects the graded spine, improves demo reliability, or addresses a
  measured failure.
- Keep the eval corpus frozen. Production corpus growth must never mutate deterministic eval results.
- Prefer seams over services: introduce interfaces and config before adding distributed systems.
- No agentic loop, router, cache, queue, or new backend without an eval, latency, or operational reason.
- Every production-sounding claim must say whether it is implemented now or reserved as a future scale
  path.
```

## Future Growth Path

### 0 to 1,000 users

Use:

- one FastAPI service
- SQLite
- Chroma
- local `/data/` or Hugging Face persistent storage
- single-worker deployment
- usage dashboard for observability

This is enough for the current course submission.

### 1,000 to 10,000 users

Consider:

- Pinecone as the default vector store
- managed Postgres for users, documents, invites, and usage logs
- object storage for uploaded files
- exact-query cache for repeated course questions
- background ingest job for large file batches
- stronger deploy monitoring

This is the first real production step.

### 10,000+ users or many cohorts

Consider:

- Redis cache
- queue-backed ingestion
- separate read/write workers
- centralized logs and metrics
- vector namespace strategy by cohort or content version
- more formal eval CI
- model/cost routing if spend becomes material

This should stay future-facing for now.

## Agentic RAG Decision

The parent notes describe query analysis, corrective retrieval, response reflection, and multi-agent
collaboration. Those are useful production patterns, but they are not justified for this Week 2
project today.

Adopt only the pieces already aligned with measured failures:

- document relevance gating through the existing answerability grader
- refusal instead of unsupported answers
- optional rerank for ranking quality
- eval-driven chunking experiments
- source provenance and faithfulness checks

Defer:

- query planners
- multi-hop decomposition
- web-search fallback
- response reflection loops
- multi-agent collaboration
- model routers

Rule of thumb: add an agentic component only when a named eval failure cannot be addressed with
simpler retrieval, chunking, or prompting changes.

## Recommended Next Work

1. Finish the Docker/Hugging Face Space deploy slice.
2. Add the system-design addendum to project docs if reviewers agree.
3. Use the new `data/` files only as a production/demo corpus, not eval data.
4. Preserve current defaults:
   - fixed chunking
   - rerank disabled unless explicitly enabled
   - local deterministic eval
   - refusal-first answer behavior
5. Final submission should explicitly say:
   "This is scale-ready by seams and evaluation discipline, not overbuilt with premature distributed
   infrastructure."

## Review Questions

Reviewers should focus on:

- Does this posture fit the one-week course submission constraint?
- Are any proposed system-design additions still too much for the timebox?
- Should the constitution addendum be added to `AGENTS.md`, `specs/roadmap.md`, or a separate docs
  file?
- Is the deploy slice the right next implementation step, or should final project packaging come
  first?
