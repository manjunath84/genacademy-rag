# Handoff: Enable cross-encoder rerank in the HF Space + faster Nebius generation model

**Date:** 2026-06-11
**Audience:** an autonomous coding agent (e.g. a Codex `/goal` run). This document is
self-contained; no access to the originating conversation is assumed.
**Repo:** `github.com/manjunath84/genacademy-rag`, local at
`/Users/manjunathans/projects/GenAcademy/Week2-RAG_ContextEngineering/genacademy-rag`
**Read first:** `AGENTS.md` (non-negotiables), `specs/roadmap.md`, `docs/deploy.md`,
`eval/phase2-rerank-delta.md`

## Goal

Enable the cross-encoder reranker in the deployed Hugging Face Space
(`https://Manjunath84-genacademy-rag.hf.space`) to capture its measured retrieval win, and switch
the serving generation model to a faster model on the Nebius preset, keeping total request latency
comfortably under the project's 8 s ceiling.

## Context (corrected premise — important)

- Rerank has a committed, measured win: recall@k 0.67 → 0.79, precision@k 0.22 → 0.25,
  MRR 0.55 → 0.58 (`eval/phase2-rerank-delta.md`).
- Rerank is **not** disabled because the latency budget is blown (worst case ~6.7 s < 8 s). It is
  disabled because the cross-encoder model (`cross-encoder/ms-marco-MiniLM-L6-v2`) is **not baked
  into the Docker image** — the `Dockerfile` only pre-downloads the embedding model, and
  `GENACADEMY_RERANK_LOCAL_FILES_ONLY=true` (default) blocks runtime downloads.
- The real latency risk: the committed 886 ms p95 rerank retrieval latency (vs 286 ms baseline)
  was measured on local dev hardware at `rerank_pool=0` — scoring the entire fused candidate union
  (~40 chunks × ~1000 chars, fp32 CPU) inside the corpus lock
  (`src/genacademy_rag/core/retriever.py` ~lines 101–107). The Space CPU is weaker; expect 1.5–3×.

## Locked decisions (user-approved — do not relitigate)

1. **Serving stays on the Nebius preset** (`GENACADEMY_PROVIDER=nebius`) with a **faster model**
   replacing the current Llama-3.3-70B-class `NEBIUS_MODEL`. No new provider, no new API key.
2. **Full 15-question eval rerun** with the final configuration before calling it done — run on
   the Nebius preset (roadmap mandate: regenerated eval reports must run on Nebius).
3. Starting rerank pool: `GENACADEMY_RERANK_POOL=20` (default `0` = full union), subject to the
   recall gate in Step 3.

## File map

| What | Where |
| --- | --- |
| Rerank implementation | `src/genacademy_rag/core/reranker.py` (`build_reranker`); eager-loaded at startup |
| Rerank integration | `src/genacademy_rag/core/retriever.py` ~lines 101–107 — reranks after RRF fusion, inside the corpus lock, truncates to `top_k=5` |
| Config (all env-driven; no code change needed) | `src/genacademy_rag/config.py` — `GENACADEMY_RERANK_ENABLED` (default false), `GENACADEMY_RERANK_POOL` (default 0), `GENACADEMY_RERANK_LOCAL_FILES_ONLY` (default true), `GENACADEMY_RERANK_BATCH_SIZE`, `NEBIUS_MODEL`, `PROVIDER_PRESETS` |
| Model provisioning script (already exists) | `scripts/provision_rerank_model.py` — downloads the cross-encoder into `HF_HOME` |
| Docker gap | `Dockerfile` — bakes the embedding model only; `HF_HOME=/app/.cache/huggingface` |
| Request path (2 LLM calls) | `src/genacademy_rag/core/graph.py` — JSON-mode grader (max_tokens=64, `grader.py`; cosine-threshold fallback on JSON failure) → answer (max_tokens=800) or refuse |
| Latency table | `docs/deploy.md` § "Rerank Demo Toggle Recommendation": baseline p95 286 ms → rerank p95 886 ms |
| Live smoke procedure | `docs/deploy.md` § "Live Acceptance-Test Order" (9 steps) |
| Latency observability | `usage_log.latency_ms`, surfaced on `/admin/dashboard` |

## Implementation steps (bottom-up; each gate must pass before the next step)

### Step 1 — Bake the rerank model into the Docker image

In `Dockerfile`, after the embedding pre-download step, add:

```dockerfile
# Pre-download the cross-encoder rerank model so GENACADEMY_RERANK_ENABLED=true works offline.
RUN uv run --no-sync python scripts/provision_rerank_model.py
```

Image grows ~90 MB — acceptable.
**Gate:** local `docker build` succeeds; container boots with `GENACADEMY_RERANK_ENABLED=true`
performing **no runtime model download**.

### Step 2 — Benchmark and choose the faster Nebius model (gated)

Check the current Nebius Token Factory catalog; benchmark 2–3 candidates (e.g. a
Llama-3.1-8B-fast-class model and a Qwen-32B-fast-class model). For each:

- **Latency:** 5 warm generation calls at the answer-call shape (~800 max_tokens). Target < 2 s.
- **JSON-mode grader gate (load-bearing):** ~10 grader-shaped JSON-mode calls (max_tokens=64).
  **Require 10/10 clean parses.** A model that flubs JSON silently degrades every request to the
  cosine-threshold fallback — the refusal path is the product promise; do not ship a model that
  fails this gate.

Selection rule: fastest model that passes the JSON gate. If the 8B-class model fails JSON or
craters faithfulness in Step 3, use the 32B-fast-class model.

### Step 3 — Full eval rerun with the final config (gated)

Run the deterministic 15-question eval (see `eval/` for the runner) with:

```
GENACADEMY_PROVIDER=nebius NEBIUS_MODEL=<chosen>
GENACADEMY_RERANK_ENABLED=true GENACADEMY_RERANK_POOL=20
```

- Regenerate `eval/REPORT.md` (must be on the Nebius preset — roadmap mandate).
- Extend the delta table in `eval/phase2-rerank-delta.md`: baseline / rerank pool=0 /
  rerank pool=20, with retrieval metrics **and** latency columns.
- **Recall gate:** recall@k at pool=20 must hold ≥ ~0.79. If the pool cap costs recall, try
  pool=30; else revert to pool=0 and accept the latency.
- **Faithfulness gate:** if faithfulness drops materially below the 58% baseline, switch to the
  larger fast model (Step 2 fallback).

### Step 4 — Space configuration + deploy (human-in-the-loop)

Update HF Space variables (no code): `NEBIUS_MODEL=<chosen>`, `GENACADEMY_RERANK_ENABLED=true`,
`GENACADEMY_RERANK_POOL=<settled value>`; keep `GENACADEMY_RERANK_LOCAL_FILES_ONLY=true`.
Push `main` → Space rebuild picks up the new Dockerfile layer.

**Stop and hand the user the exact variable list unless you hold an HF token with write access to
the Space — do not guess credentials.**

**Rollback / kill switch:** `GENACADEMY_RERANK_ENABLED=false` (or reverting `NEBIUS_MODEL`) in
Space variables restarts the app **without a rebuild**. Document this in `docs/deploy.md`.

### Step 5 — Live validation + docs (gated)

- Run the full 9-step Live Acceptance-Test Order in `docs/deploy.md`.
- Ask ~10 questions; read `latency_ms` p95 from `/admin/dashboard`.
  **Gate: p95 < 8 s hard, ~6 s goal.** If Space-CPU rerank blows it: drop pool to 10 and
  re-measure; kill switch as last resort.
- Then update docs (only after live numbers exist):
  - `docs/deploy.md` — env table (new model, rerank vars true), rerank section with live numbers +
    kill-switch note.
  - `docs/project-writeup.md` — remove/update the limitation "Rerank is disabled in the Space
    because the rerank model is not baked into the Docker image"; refresh cited eval numbers.
  - `README.md` if it mentions rerank being disabled.

## Binding rules (from `AGENTS.md` / `CLAUDE.md`)

- **Branch off `main`; never commit directly to it.** (PR #14 `feat/compass-house-theme` may still
  be open — branch from `main` after it merges, or independently.)
- **Builder ≠ reviewer:** an independent model/fresh context reviews the PR before merge.
- **Evidence before "done":** show `uv run ruff check .` + `uv run pytest -q` output (baseline
  265 passed as of 2026-06-11; no `src/` changes expected — regression check), the eval tables,
  and the live latency numbers. "It should work" does not count.
- **Never ingest the `Mastering-Agentic-AI-Week2` sample-solution repo.**
- **Do not move rerank out of the corpus lock** — the lock deliberately spans two consistency
  domains (Chroma + in-memory BM25; prior review finding). Serializing concurrent asks is accepted
  at cohort traffic.

## Prerequisites (stop and ask if missing)

- `NEBIUS_API_KEY` in the environment (Steps 2–3 are live calls; the 15-question eval spends
  tokens on grader + answer + judge calls).
- Docker available locally (Step 1 verification).
- HF Space access for Step 4 (otherwise hand the variable list to the user).

## Acceptance criteria (definition of done — all required, with evidence)

1. Local `docker build` succeeds; container boots rerank-enabled with no runtime model download.
2. Eval tables committed: recall@k ≥ 0.79 at the shipped pool; faithfulness not materially < 58%;
   `eval/REPORT.md` regenerated on the Nebius preset.
3. `ruff` clean and `pytest` green (regression check).
4. Live Space: 9-step acceptance order passes; dashboard p95 within budget; cited answer and
   refusal both verified in a browser.
5. Docs updated with live numbers; independent review done; PR merged.

If a gate fails and both of its listed fallbacks fail, **stop and report** rather than improvising
a third option — the fallback chains were deliberately bounded.
