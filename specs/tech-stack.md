# Tech Stack

*Canonical. Layers + **binding guardrails** (a reviewer rejects violations). Status: review incorporated
(Kimchi, 2026-06-07), plan-ready pending spike. Deep reasoning: `../docs/architecture-decisions.md`.*

## Layers

```
FastAPI (one service)
├── View:  HTMX + Tailwind + Alpine.js   — Phase 0: NON-STREAMING form-post; SSE optional (first to cut)
├── Auth:  server sessions + role gate   — admin / member  (P0: 2 seeded users; P1: invite-code RBAC)
├── Core (pure, no web imports):
│   ├── Loader registry  — PDF, DOCX (P0) → web (P1)
│   ├── Chunker iface     — FixedSizeChunker ~512/64 (P0) → SectionAwareChunker (P2)
│   ├── ModelProvider    — embed: local sentence-transformers all-MiniLM-L6-v2 (384-dim, P0)
│   │                      generate: Nebius (MANDATORY ≥1 call)
│   ├── VectorStore iface — Chroma (P0) → Pinecone (P2)
│   ├── Retriever  iface  — Hybrid dense+BM25+RRF, top_k=5 (P0) → + cross-encoder rerank (P2)
│   └── LangGraph graph   — retrieve → grade → {answer + citations | refuse}
└── Datastore iface — SQLite (P0) → Postgres (deploy)
    └── tables: users · documents · chunks_meta   (usage_log → P1)
```

## Language & tooling

- **Python 3.12**, **`uv`** (lockfile committed), **`ruff`**, **`pytest`**.
- **LangChain** primitives + **one** small **LangGraph** graph (refusal branch only); linear steps stay
  LCEL.
- `rank-bm25` for the sparse half of hybrid; `sentence-transformers` for local embeddings.
- OpenAI-compatible SDK pointed at **Nebius** `base_url` for **generation** (the mandatory call).
- Vectors in Chroma/Pinecone; **everything else** (users, doc metadata; usage in P1) in `Datastore`.

## Binding guardrails (review-blockers)

1. **Pure core / thin view.** No `fastapi` / template imports inside `core/`.
2. **Pluggability = interface + config preset, not scattered conditionals.** New provider/store/chunker
   = a new class + a config entry. No `if provider == ...` in business logic.
3. **Citation metadata captured at ingest** and threaded to the answer. Never reconstructed post-hoc.
4. **Refusal path cannot be bypassed** to improve a demo. Low confidence → refuse, don't answer from
   priors.
5. **Reference calls verbatim.** Nebius model IDs, embedding **dimension (384)**, Pinecone index config,
   API/request schemas, and the **LLM-judge + grader prompts** are **copied from source** into
   `specs/<feature>/requirements.md` — never paraphrased. Embedding dim MUST match the vector index dim.
6. **Pin LangChain/LangGraph to exact versions** in `pyproject.toml` (mid-week releases break pipelines).
7. **Secrets via env only.** Nebius/Pinecone keys never committed, never logged.

## Grader mechanism (decided; spike-gated)

Primary: **Nebius JSON-mode LLM grader** → `{"answerable": bool, "confidence": 1-5}`, fast/cheap model,
~500 ms budget. Fallback (if JSON mode absent or latency blows the < 8 s ceiling): **max cosine
similarity** of query vs top-k chunks against a **calibrated threshold** (tuned on 3–5 held-out Qs).

## Unverified specifics (resolve in the spike before locking)

- Nebius **chat model ID** + **JSON mode / structured-output** support (gates grader + LLM-judge format).
- **Throughput / rate limits** (eval = 15 generate + 15 judge calls in a loop).
- Measured **latency** (local embed + Nebius generate) vs the < 8 s ceiling.
- **20 MB guidebook parse quality** → OCR fallback / exclude-if-bad.
- Pinecone free-tier index config (dimension must match **384**).

## Reproducible deploy (Phase 2)

Lockfile → pinned deps → install in a **clean** env → Docker image → **smoke-check the live URL** (one
real query returns a cited answer) **before** announcing.
