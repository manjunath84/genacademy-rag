# Tech Stack

*Canonical. Layers + **binding guardrails** (a reviewer rejects violations). Status: review incorporated
(Kimchi, 2026-06-07); spike complete (2026-06-08, `../docs/spike-findings.md`). Deep reasoning:
`../docs/architecture-decisions.md`.*

## Layers

```
FastAPI (one service)
├── View:  HTMX + Tailwind + Alpine.js   — Phase 0: NON-STREAMING form-post; SSE optional (first to cut)
├── Auth:  server sessions + role gate   — admin / member  (P0: 2 seeded users; P1: invite-code RBAC)
├── Core (pure, no web imports):
│   ├── Loader registry  — GitHub fetcher + Markdown/Jupyter (P0: commit-pinned EVAL corpus)
│   │                      + PDF/DOCX (PRODUCTION files) → Pptx/JSON/Python/web (later, as corpus grows)
│   ├── Chunker iface     — FixedSizeChunker ~512/64 (P0) → SectionAwareChunker (P2)
│   ├── ModelProvider    — embed: local sentence-transformers all-MiniLM-L6-v2 (384-dim, P0)
│   │                      generate: OpenAI-compat presets — Nebius (MANDATORY for submission) /
│   │                      OpenRouter (dev default) / OpenAI / local Gemma — one seam, config-only swap
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
   GitHub sources carry `{repo, file_path, line_start/end, commit_hash}`; file sources carry
   `{doc_id, title, page/section, char_span}`.
4. **Refusal path cannot be bypassed** to improve a demo. Low confidence → refuse, don't answer from
   priors.
5. **Reference calls verbatim.** Nebius model IDs, embedding **dimension (384)**, Pinecone index config,
   API/request schemas, and the **LLM-judge + grader prompts** are **copied from source** into
   `specs/<feature>/requirements.md` — never paraphrased. Embedding dim MUST match the vector index dim.
6. **Pin LangChain/LangGraph to exact versions** in `pyproject.toml` (mid-week releases break pipelines).
7. **Secrets via env only.** Nebius/Pinecone keys never committed, never logged.
8. **Eval corpus is commit-pinned; the `Mastering-Agentic-AI-Week2` notebooks/code are never ingested or
   read** (the sample solution — reading it to inform the build is disqualifying). Eval = one frozen
   snapshot + one gold set; production tracks HEAD + uploads and never expands the gold set.

## Grader mechanism (decided; spike-confirmed)

Primary: **JSON-mode LLM grader** → `{"answerable": bool, "confidence": 1-5}`, fast/cheap model,
~500 ms budget. **Spike confirmed JSON mode works on an open Llama model** (via OpenRouter; Nebius
serves the same model class). Fallback (JSON mode absent or latency blows the < 8 s ceiling): **max
cosine similarity** of query vs top-k chunks against a **calibrated threshold** (tuned on held-out Qs).

## Spike-verified specifics (2026-06-08, `../docs/spike-findings.md`)

- ✅ **JSON mode / structured output** — works on open Llama 3.1 70B (grader + LLM-judge format unblocked).
- ✅ **Throughput** — 10/10 sequential calls, no throttling, on two providers.
- ✅ **Latency** — embed 12 ms + generate ~4 s ≪ the < 8 s ceiling.
- ✅ **GitHub fetch + commit-pin** — clean; gold-set SHAs pinned in the findings.
- ✅ **Guidebook parse** — pypdf adequate (0.994 printable ratio); **no OCR needed**.
- ✅ **Nebius confirm** — credit landed; the graded eval + model-swap demo ran on the Nebius preset
  (`meta-llama/Llama-3.3-70B-Instruct`, `eval/REPORT.md`, commit `7c85f81`). Dev default stays OpenRouter.
- 🟢 **Pinecone free-tier config** (dim must match **384**) — deferred to Phase 2 start.

## Reproducible deploy (Phase 2)

Lockfile → pinned deps → install in a **clean** env → Docker image → **smoke-check the live URL** (one
real query returns a cited answer) **before** announcing.
