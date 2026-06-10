# GenAcademy RAG Phase 2 Nebius Embeddings Preset Implementation Plan

**Goal:** the roadmap's "Nebius embeddings preset — the 'swap embedding provider' demo".
`GENACADEMY_EMBEDDINGS=nebius` is the one config line; default `local` is byte-identical.

**Approval note:** pre-approved workflow (fix → review → PR); plan committed as the audit artifact.

## Scope

- `OpenAICompatEmbedder` in `core/providers.py`: OpenAI-compatible `/embeddings` endpoint
  (Nebius AI Studio is one), L2-normalized output (matches `STEmbedder`'s
  `normalize_embeddings=True`, which the grader's cosine threshold assumes), input batching,
  response items re-ordered by `.index`.
- `build_embedder(settings)` factory; `build_provider` routes through it. Default `local`
  constructs `STEmbedder` exactly as before.
- Settings: `GENACADEMY_EMBEDDINGS` (`local`|`nebius`, validated eagerly like the provider and
  vectorstore), `NEBIUS_EMBED_MODEL` (default `Qwen/Qwen3-Embedding-8B`), reusing
  `NEBIUS_BASE_URL`/`NEBIUS_API_KEY` independently of the generation provider — so
  `GENACADEMY_PROVIDER=openrouter` + `GENACADEMY_EMBEDDINGS=nebius` works.
  `GENACADEMY_EMBED_DIM` (default 384) replaces the Pinecone factory's hardcoded constant —
  a remote-embedding corpus needs a matching-dimension index.
- Missing key fails loudly (`EmbedderSetupError`), never a silent local fallback.

## Invariants

- **Eval stays local and offline.** `scripts/eval_retrieval.py` and `run_eval.py` construct
  `STEmbedder` directly and are untouched. The deterministic retrieval eval never gains a
  network dependency.
- **Same-embedder corpus rule:** query and ingest must use the same embedder. Switching
  `GENACADEMY_EMBEDDINGS` against an existing collection fails loudly at query time
  (dimension mismatch from the store), never silently: the documented demo path is
  re-ingesting into a fresh collection with the env set.
- **Offline tests:** fake OpenAI client injected (same pattern as the reranker's
  `cross_encoder_cls`); no live calls in pytest. Live smoke script verifies the real
  endpoint (key exists in `.env`).

## Tasks

1. Settings + eager validation + tests.
2. `OpenAICompatEmbedder` + `build_embedder` + tests (TDD, fake client).
3. Wire `build_provider`; Pinecone factory uses `settings.embed_dim`; tests.
4. `.env.example`; `scripts/smoke_nebius_embeddings.py` live smoke (no secrets printed).
5. Evidence: ruff + pytest + live smoke; codex review before PR; merge.
