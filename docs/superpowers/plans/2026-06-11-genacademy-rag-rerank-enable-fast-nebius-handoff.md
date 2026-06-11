# Handoff prompt — paste this into the implementing agent

---

You are implementing an already-approved plan in the repo `manjunath84/genacademy-rag`
(GenAcademy RAG — FastAPI + hybrid retrieval + Nebius generation, deployed as a Docker HF Space).
Work on branch `claude/refine-local-plan-if1rmg` (exists locally and on origin, based on `main`
@ `c66fe22`). The working tree is clean — nothing from the planning session was committed.

Read `AGENTS.md` first and follow it: evidence before "done" (show `ruff` + `pytest` output),
builder ≠ reviewer (the PR gets an independent review pass), refusal path is load-bearing.

**Environment notes from the planning session (verified, don't re-litigate):**
- The plan below was verified line-by-line against the codebase on 2026-06-11; the file/line
  references are accurate as of `c66fe22`.
- If you are running in a sandbox whose network policy blocks `huggingface.co` (the planning
  sandbox got `403 host_not_allowed`), Part A steps A2 (model provisioning + eval) and A3
  (docker build) cannot run there — they need a machine with HF access (e.g. the user's laptop).
  Everything else in Part A is pure file edits and runs anywhere. `raw.githubusercontent.com`
  was reachable, so the eval-corpus fetch itself is not the blocker.
- No `NEBIUS_API_KEY` is needed for anything in Part A. Part B is an operator runbook that
  needs the key and HF Space access — you write it into `docs/deploy.md`; the user executes it.

Implement **Part A** of the plan below, in order, gating as specified. Commit with clear
messages and push with `git push -u origin claude/refine-local-plan-if1rmg`. Do not create a
PR unless the user asks. Do not fabricate eval numbers: if you cannot run A2/A3 in your
environment, commit the doc/Dockerfile/spike changes, leave `eval/phase2-rerank-delta.md`
untouched, make sure the A2 commands + gates appear verbatim in the deploy.md runbook so the
user can run them locally, and say clearly in your final report which steps remain.

---

# THE PLAN

# Enable rerank in the Space, offset latency with a faster Nebius model

## Context

The cross-encoder reranker has a committed, measured win — recall@k 0.67 → 0.79, MRR 0.55 → 0.58
(`eval/phase2-rerank-delta.md`) — but is disabled in the deployed HF Space. The actual blocker is
that the rerank model is **not baked into the Docker image** (`Dockerfile:24-25` only pre-downloads
`all-MiniLM-L6-v2`; `GENACADEMY_RERANK_LOCAL_FILES_ONLY` defaults `true` in `config.py:155`, blocking
runtime downloads — confirmed at `docs/deploy.md:539-540`). The latency concern is still real: the
886ms rerank p95 was measured at `rerank_pool=0` (scores the entire fused union — `retriever.py:102`)
on local hardware; the Space CPU is weaker. Mitigations: cap the pool (`GENACADEMY_RERANK_POOL=20`)
and switch generation to a faster Nebius model (2 LLM calls per request: grader + answer dominate
the budget).

**Locked decisions:** stay on the Nebius preset with a faster model (current:
`meta-llama/Llama-3.3-70B-Instruct`, `specs/roadmap.md:114`); validate with a full eval rerun.

**Execution split:** Part A = key-free repo work (this PR). Part B = operator runbook (needs
Nebius key / Space access), shipped inside the PR's deploy.md.

No `src/` changes — the seams already exist (`build_reranker(settings)` in `core/reranker.py:87`,
`rerank_pool` plumbed through `web/app.py:496-497`, `scripts/run_eval.py:39-40`, and
`scripts/eval_retrieval.py:79-80`; all env-driven via `config.py:150-159`).

```mermaid
graph TD
    subgraph A["Part A — repo work (no keys) → PR"]
        A1[Dockerfile: bake rerank model] --> A3[docker build +<br/>offline-load proof]
        A2[eval_retrieval.py: baseline vs<br/>pool=0 vs pool=20] --> G1{recall@k ≥ 0.79<br/>at pool=20?}
        G1 -- yes --> A4[extend phase2-rerank-delta.md<br/>+ doc updates + runbook]
        G1 -- no --> A2b[try pool=30, else ship pool=0] --> A4
        A3 --> A5[ruff + pytest, push branch]
        A4 --> A5
    end
    subgraph B["Part B — operator (Nebius key + Space access)"]
        B1[gen_probe.py per candidate model] --> G2{JSON gate 10/10<br/>+ fastest?}
        G2 --> B2[full run_eval.py rerun<br/>→ eval/REPORT.md] --> G3{faithfulness<br/>not < 58%?}
        G3 -- no --> B1
        G3 -- yes --> B3[Space vars + push main] --> B4[live acceptance tests<br/>+ dashboard p95 gate]
        B4 -- p95 over budget --> B5[pool 20→10 → kill switch]
        B4 -- pass --> B6[commit live numbers<br/>into deploy.md]
    end
    A5 --> B1
```

## Part A — this PR (branch `claude/refine-local-plan-if1rmg`)

### A1. Bake the rerank model into the Docker image

`Dockerfile` — after the embedding pre-download (lines 23-25), add:

```dockerfile
# Pre-download the cross-encoder rerank model so GENACADEMY_RERANK_ENABLED=true works offline.
RUN uv run --no-sync python scripts/provision_rerank_model.py
```

Reuses `scripts/provision_rerank_model.py` as-is (the project's only sanctioned network path for the
reranker: `local_files_only=False`, downloads `cross-encoder/ms-marco-MiniLM-L6-v2` into
`HF_HOME=/app/.cache/huggingface`). It runs fine at build time: `Settings.from_env()` needs no keys,
the layer runs as `user` after the `chown` (Dockerfile:20-21), and `rerank_cache_dir=None` falls back
to `HF_HOME`. Image grows ~90MB — acceptable.

### A2. Deterministic recall gate at pool=20 (no key needed; needs huggingface.co access)

The deterministic retrieval eval is key-free (`scripts/eval_retrieval.py` docstring: "No LLM, no
generation key"). Run:

```bash
uv run python scripts/provision_rerank_model.py          # one-time local cache
uv run python scripts/ingest_eval_corpus.py              # pinned corpus → Chroma 'eval'
# three runs on the SAME box so latency columns are apples-to-apples:
GENACADEMY_RERANK_ENABLED=false uv run python scripts/eval_retrieval.py --collection eval
GENACADEMY_RERANK_ENABLED=true  GENACADEMY_RERANK_POOL=0  uv run python scripts/eval_retrieval.py --collection eval
GENACADEMY_RERANK_ENABLED=true  GENACADEMY_RERANK_POOL=20 uv run python scripts/eval_retrieval.py --collection eval
```

(`eval_retrieval.py` already reports per-question `retrieval_ms` and writes a config snapshot
including `rerank_pool` via `--json-out` — nothing new to build.)

**Gate:** recall@k at pool=20 must hold ≥ 0.79 and MRR ≥ ~0.58. If the cap costs recall, try
pool=30; if that still loses, ship pool=0 and accept the latency. Extend
`eval/phase2-rerank-delta.md` with a pool-comparison section: baseline / pool=0 / pool=20 rows,
retrieval metrics + latency, dated, with an explicit note that this run is on different hardware
than the 2026-06-09 run (hence the same-box pool=0 re-run for the latency comparison).

If your environment blocks huggingface.co: skip A2 here, keep the commands above verbatim in the
deploy.md runbook (A4) for the user to run locally, and do NOT invent numbers.

### A3. Prove the bake (needs docker + huggingface.co access)

```bash
docker build -t genacademy-rag:rerank .
docker run --rm --network=none -e GENACADEMY_RERANK_ENABLED=true genacademy-rag:rerank \
  uv run --no-sync python -c "from genacademy_rag.config import Settings; \
from genacademy_rag.core.reranker import build_reranker; \
assert build_reranker(Settings.from_env()) is not None; print('rerank model loads offline')"
```

`--network=none` is the proof: with `local_files_only=true` (default) and no network, a successful
load means the model is in the image. (A failed bake raises `RerankerUnavailableError`,
`core/reranker.py:65-67`.) If blocked in your environment, include this as a verification step in
the runbook instead.

### A4. Doc updates that don't depend on live numbers

- `docs/deploy.md`:
  - Variables block (line 67): `GENACADEMY_RERANK_ENABLED=true`, add `GENACADEMY_RERANK_POOL=20`.
  - Acceptance-test step 1 (lines 273-276): expectation flips to `GENACADEMY_RERANK_ENABLED=true`
    (+ pool var).
  - "Rerank Demo Toggle Recommendation" (lines 466-488): rewrite — rerank is now default-on in the
    Space (model baked); add the **kill switch** (flipping `GENACADEMY_RERANK_ENABLED=false` or
    reverting `NEBIUS_MODEL` in Space variables restarts without a rebuild); add pool=20 numbers
    from A2 if available; leave a clearly-marked placeholder row for live Space p95 (filled in B6).
  - "Known Restrictions" (lines 539-540): replace "rerank model is not baked" with "rerank model is
    baked into the image; rebuild when changing `GENACADEMY_RERANK_MODEL`".
  - **New "Enabling Rerank + Model Swap Runbook" subsection** = Part B below, so the operator steps
    ship with the PR.
- `docs/project-writeup.md:201`: remove/update the limitation "Rerank is disabled in the Space
  because the rerank model is not baked into the Docker image".
- `README.md`: **no change** — it never claims rerank is Space-disabled; line 45's
  `GENACADEMY_RERANK_ENABLED=false` eval command is the deterministic local default and stays
  (the code default is still off; only the Space turns it on).
- `eval/REPORT.md` is **not** touched in Part A — it's regenerated only by the full keyed rerun
  (B2), keeping the Nebius-preset mandate (`specs/roadmap.md:111-115`) intact.
- `spike/gen_probe.py`: strengthen the JSON check from a single call to **10 grader-shaped calls**
  (`json_mode` / `response_format={"type": "json_object"}`, `max_tokens=64`, the
  `{"answerable": <bool>, "confidence": <1-5 int>}` shape from `grader.py:18-24`); `json_ok`
  requires 10/10 clean parses. Spike code is throwaway — keep the edit minimal and in the file's
  existing style.

### A5. Regression check + push

`uv run ruff check .` and `uv run pytest -q` (no src changes expected — pure regression check).
Commit and push to `claude/refine-local-plan-if1rmg`
(`git push -u origin claude/refine-local-plan-if1rmg`). Per AGENTS.md, the PR gets an independent
review pass before merge (builder ≠ reviewer).

## Part B — operator runbook (needs `NEBIUS_API_KEY` / Space access; written into deploy.md by A4)

### B1. Pick the faster Nebius model — reuse `spike/gen_probe.py`

Per model it measures single-call latency, JSON-mode parse (now 10 grader-shaped calls after A4),
and 10-sequential-call throughput/throttling. Run once per candidate:
`NEBIUS_API_KEY=... NEBIUS_MODEL=<candidate> uv run python spike/gen_probe.py`.
Check the current Token Factory catalog for candidates (a Llama-3.1-8B-fast-class and a
Qwen-32B-fast-class model are the expected shapes).

**Selection rule:** fastest model that passes the JSON gate (10/10 parses); if the 8B-class fails
JSON or craters faithfulness in B2, fall back to the 32B-fast-class. A model that flubs JSON
silently degrades every request to the cosine-threshold fallback (`grader.py:58-61`) — degraded
mode, not a plan.

### B2. Full eval rerun with the final config

```bash
GENACADEMY_PROVIDER=nebius NEBIUS_API_KEY=... NEBIUS_MODEL=<chosen> \
GENACADEMY_RERANK_ENABLED=true GENACADEMY_RERANK_POOL=20 \
uv run python scripts/run_eval.py
```

Regenerates `eval/REPORT.md` (Nebius preset ✓). **Gates:** recall@k matches the A2 result;
faithfulness not materially below 58% (`eval/REPORT.md:12`) — if it drops, switch to the larger
fast model (B1 fallback). Commit the regenerated report (same review rule).

### B3. Space configuration + deploy

Space **Variables** (no code): `NEBIUS_MODEL=<chosen>`, `GENACADEMY_RERANK_ENABLED=true`,
`GENACADEMY_RERANK_POOL=20`; keep `GENACADEMY_RERANK_LOCAL_FILES_ONLY=true`. Then push `main` to
the `hf` remote → rebuild picks up the new Dockerfile layer. **Kill switch:** flip
`GENACADEMY_RERANK_ENABLED=false` (or revert `NEBIUS_MODEL`) in Space variables — restart only,
no rebuild.

### B4. Live validation

- Run the 9-step **Live Acceptance-Test Order** (`docs/deploy.md:268-291`).
- Ask ~10 questions; read `latency_ms` p95 from `/admin/dashboard` (usage_log already records it —
  `web/app.py:215-225`, `core/analytics.py:35`). **Gate:** p95 < 8s with margin (~6s target).
- Fallback ladder if over budget: `GENACADEMY_RERANK_POOL=10` and re-measure → kill switch last.

### B5/B6. Fill the live numbers

Replace the deploy.md placeholder with measured Space p95; refresh eval numbers cited in
`docs/project-writeup.md` if changed.

## Files touched

| File | Change | Part |
|---|---|---|
| `Dockerfile` | +1 RUN line after line 25 (provision rerank model) | A |
| `eval/phase2-rerank-delta.md` | + pool-comparison section (only if A2 ran — never fabricate) | A |
| `docs/deploy.md` | env block, acceptance step 1, rerank-toggle rewrite + kill switch, known restrictions, new operator runbook | A |
| `docs/project-writeup.md` | drop "not baked" limitation (line 201) | A |
| `spike/gen_probe.py` | strengthen JSON check to 10 grader-shaped calls | A (used in B1) |
| `eval/REPORT.md` | regenerated by full keyed rerun | B |
| HF Space variables | `NEBIUS_MODEL`, `GENACADEMY_RERANK_ENABLED=true`, `GENACADEMY_RERANK_POOL=20` | B |

## Risks & notes

- **Space CPU rerank latency unknown** — 886ms p95 was local at pool=0. Mitigated by pool=20, A2's
  same-box comparison, B4's live gate, and the no-rebuild kill switch.
- **A2 hardware ≠ the 2026-06-09 run's hardware** — that's why A2 re-runs pool=0 on the same box
  and the new delta section is labeled accordingly.
- **Rerank runs inside the corpus lock** (`retriever.py:101-107`), serializing concurrent `/ask`.
  Acceptable at cohort traffic; documented, not changed (prior review finding).
- **Eval mandate preserved**: `eval/REPORT.md` only ever regenerated on the Nebius preset (B2).

## Verification

1. **A3 offline-load proof**: `docker run --network=none ... build_reranker(...)` prints
   `rerank model loads offline`.
2. **A2 gate**: recall@k ≥ 0.79 at the shipped pool setting; extended delta table committed.
3. `uv run ruff check .` + `uv run pytest -q` green.
4. **B gates** (operator): 10/10 JSON parses on the chosen model; faithfulness not materially
   < 58%; live acceptance order passes; dashboard p95 < ~6s; cited answer + refusal verified in
   browser.
