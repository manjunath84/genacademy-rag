# GenAcademy RAG Phase 2 Pinecone Preset Implementation Plan

**Goal:** the roadmap's "Pinecone preset — second `VectorStore` impl; live 'Chroma → Pinecone, one
config line'" slice. `GENACADEMY_VECTORSTORE=pinecone` is the one config line.

**Approval note:** the user pre-approved this slice ("don't ask for approval until you create the
next PR"); the plan is committed as the audit artifact rather than gated.

## Scope

- `PineconeStore` implementing the existing `VectorStore` Protocol (`upsert`, `query`, `get_chunk`,
  `get_all_chunks`, `delete_doc`) in `core/vectorstore.py`.
- `build_vectorstore(settings, *, collection)` factory in the same module; `collection` maps to a
  Pinecone namespace.
- Settings: `GENACADEMY_VECTORSTORE` (default `chroma`), `PINECONE_API_KEY` (env-only secret),
  `GENACADEMY_PINECONE_INDEX`, `GENACADEMY_PINECONE_CLOUD`, `GENACADEMY_PINECONE_REGION`.
  New fields carry safe defaults so the chroma path is byte-identical with no env changes.
- Web wiring: only the **serving** store routes through the factory. The `eval` collection and all
  eval/ingest scripts stay pinned to local Chroma — the deterministic-eval invariant is not
  negotiable, and a remote store would put network state inside the protected metric.
- `pinecone` SDK added as a pinned dependency; imported lazily inside `PineconeStore.__init__`.

## Fixed decisions

- **Classic client API** (`Pinecone(api_key)`, `has_index`, `create_index(spec=ServerlessSpec)`,
  `Index(name)`) — retained as compatibility shims through SDK v9, so the wrapper survives SDK
  major bumps.
- **Score semantics:** Pinecone's cosine `match.score` IS similarity (unlike Chroma's distance), so
  it passes through without the `1 - d` conversion. `RetrievedChunk.score` stays cosine either way.
- **Chunk text lives in vector metadata** (`text` key) plus `ordinal`; `Citation.to_metadata()`
  already strips `None`s (Pinecone rejects null metadata). Numeric metadata returns as float from
  Pinecone's JSON — reconstruction coerces `ordinal`/`line_*`/`char_*` back to int.
- **`delete_doc` deletes by id-prefix listing** (`{doc_id}::`), not metadata filter — serverless
  indexes don't support filtered deletes.
- **`get_all_chunks` sorts by `(doc_id, ordinal)`** — Pinecone list order is arbitrary, and the
  BM25 index build needs a deterministic corpus order.
- **Missing key fails loudly:** `GENACADEMY_VECTORSTORE=pinecone` without `PINECONE_API_KEY` raises
  `VectorStoreSetupError` at startup, never a silent chroma fallback.
- **Tests are offline:** a fake Pinecone client/index records calls; no real SDK network calls.
  Live smoke against a real index is a documented manual step once a key exists.

## Known limitations (reviewed and accepted for this slice)

- **Eventual consistency:** Pinecone serverless reads lag writes. After the post-merge review
  hardening, the web app derives its in-memory corpus from the retriever's snapshot plus local
  deltas on every mutation (boot seeds from the local seed list; upload unions the committed
  chunks into the snapshot; delete filters the snapshot) — no mutation path performs a remote
  re-read, so search correctness does not depend on remote reads having caught up with any
  mutation, its own or earlier ones. The admin reindex is the one deliberate remote re-read
  (recovery path for chunks missing from memory); it is filtered against the datastore's
  deletion ledger so orphaned vectors from a lagged delete can never resurrect a deleted doc,
  and it logs the corpus-size change. A reindex during a lag window can still shrink the corpus
  (partial remote read — `get_all_chunks` logs a warning when fetch returns fewer vectors than
  listed); reindex again once the store converges.
- **Corpus lock holds network I/O:** with Pinecone, upload/delete/reindex mutations perform
  remote calls inside the retriever's corpus lock, blocking concurrent questions for the call's
  duration. Acceptable for a single-admin demo app; a lock-free snapshot swap is the future fix
  if this preset ever serves real traffic.
- **Single-process assumption:** index auto-creation does not guard the two-workers-boot race
  (both pass `has_index`, second `create_index` conflicts). The app runs single-process.

## Tasks

1. Settings + tests (`tests/test_config.py`).
2. `PineconeStore` + `build_vectorstore` + tests (`tests/core/test_vectorstore.py`), TDD.
3. Web wiring (`build_default_app` serving store) + existing web tests stay green.
4. `.env.example`, `pyproject.toml` (pinned `pinecone`), module docstring.
5. Evidence: `uv run ruff check src tests scripts` + `uv run pytest` output; fresh-context review
   before merge (builder ≠ reviewer).
