# Pre-build spike — findings (design.md §9)

Run: 2026-06-08. Probe scripts in `spike/` (throwaway; venv gitignored). This gates `writing-plans`.

## Status

| # | Spike item | Needs key? | Status | Verdict |
|---|---|---|---|---|
| 1 | Nebius chat model ID + JSON mode | Nebius | ⏳ key later today/Mon | probe staged; **not architecture-blocking** (grader fallback decided) |
| 2 | Nebius throughput / rate limits (~10 seq calls) | Nebius | ⏳ key later today/Mon | probe staged; fallback = batch/sleep in eval loop |
| 3 | End-to-end latency vs < 8 s ceiling | Nebius | 🟡 half | local embed = 12 ms; generate half pending key |
| 4 | GitHub fetch + commit-pin (eval corpus) | no | ✅ done | clean |
| 5 | Guidebook parse-quality gate (production) | no | ✅ done | **PASS — no OCR** |
| 6 | Pinecone free-tier + index dim = 384 | Pinecone | 🟢 deferred to P2 | **not a Phase-0 gate** (P0 uses Chroma); dim=384 match already confirmed |

## Nebius — requirement clarified (cohort Discord, 2026-06-07)

Cohort staff (Tanish, The Gen Academy) confirmed:
- **Credit arrives later today / Monday.** Until then, *"you're fine testing on Claude."*
- **The requirement = make ≥1 model call running an open-source model deployed on Nebius** (e.g. an
  open Llama/Qwen), **not** "routing." This **confirms the design's interpretation**: Nebius = the
  mandatory *generation* call. No design change.
- **Build-time consequence:** generation is behind `ModelProvider.generate()` already, so Phase-0 dev
  can run against **Claude or the local Gemma** interim, then point the config at the Nebius open model
  for the mandatory call once credit lands. The Nebius probe (items 1–3) then confirms JSON-mode (→ which
  grader variant) and latency — a **config choice, not a re-architecture**.

## 4. GitHub fetch + commit-pin — ✅ CLEAN

Eval corpus = two public repos under `The-Gen-Academy`, fetched via `gh api` / `raw.githubusercontent.com`. **Pin the gold set to these SHAs:**

| Repo | default branch | **pinned HEAD SHA** | Eval-relevant blobs |
|---|---|---|---|
| `awesome-agentic-ai-resources` | `main` | `5dfb8691180dc4956107e86839998ba3a2ebd94f` | `README.md` (28 KB, 114 table rows) |
| `Mastering-Agentic-AI-Week1` | `main` | `3aa31dfede8c76422be91f2ecdbc59eddc690b1d` | `Langchain Basics/Langchain_Fundamentals.ipynb` (26 KB, 25 cells), `Langchain Basics/README.md` (1.4 KB), `Langchain Basics/langchain_prompts.py` (892 B) |

- README parses as Markdown; 114 table rows (the curriculum catalog). **Catalog spans Weeks 1–7** → validates the "Week 8" unanswerable eval question; the "Covers" table column means many "what does X cover" questions are answerable *from the catalog text*, not just links.
- Notebook is valid JSON, 25 cells (markdown + code) → `nbformat`/JSON loader is trivial.
- **Eval corpus is small** (~28 KB README + ~26 KB notebook + small README + one `.py`) → gold annotation is light; genuine multi-document questions need README + the Week-1 notebook (only two substantive docs).
- `Mastering-Agentic-AI-Week2` exists (HTTP 200) but is the **sample solution — never fetched/read** (firewall held).

## 5. Guidebook parse-quality — ✅ PASS (no OCR)

`Mastering-Agentic-AI-Getting-Started-Guidebook.pdf`, **20.2 MB but only 15 pages** (size = embedded images, not scanned pages). Plain `pypdf`:

- extract time 0.3 s · 19,321 chars · mean 1,288 chars/page · **0 empty pages** · printable ratio **0.994** · 389 heading-like lines.
- **Verdict: pypdf is adequate. OCR fallback NOT needed.** The historical "#1 risk" is low-risk. (Production corpus only; does not gate the graded eval.)

## 3. Local embedding latency — 🟡 (generation half pending)

`all-MiniLM-L6-v2`: **dim = 384 confirmed** · cold load 11.6 s (one-time at startup) · **single-query embed = 12 ms** · batch ingest 377 chunks/s. Online embed is negligible against the 8 s ceiling → the budget is effectively all Nebius generation. Confirm the generate half once the key is in.

## Path note (not eval-blocking)

The design docs reference `../CuratedRAGMaterials/` (i.e. inside `Week2-RAG_ContextEngineering/`). The folder actually lives at **`GenAcademy/CuratedRAGMaterials/`** — one level higher (`../../CuratedRAGMaterials/` from `genacademy-rag/`). Production-corpus loaders must use the correct path; update the design's path reference during planning.

## Spike verdict

**Cleared to plan Phase 0.** Every remaining item is either deferred (Pinecone → P2) or externally
blocked with a decided fallback (Nebius, key later today). Nothing open changes the Phase-0 architecture.

When the Nebius key lands: `cp spike/.env.example spike/.env`, fill `NEBIUS_API_KEY`, then run
`spike/nebius_probe.py` to confirm model id + JSON-mode (→ grader variant) + latency. Pinecone probe
(`spike/pinecone_probe.py`) only when Phase 2 starts.
