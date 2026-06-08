# Pre-build spike — findings (design.md §9)

Run: 2026-06-08. Probe scripts in `spike/` (throwaway; venv gitignored). This gates `writing-plans`.

## Status

| # | Spike item | Needs key? | Status | Verdict |
|---|---|---|---|---|
| 1 | Chat model ID + JSON mode | gen | ✅ **done via OpenRouter** | **JSON mode works on an OPEN model** (Llama 3.1 70B) → grader uses JSON primary path. Nebius re-point pending credit (low-risk confirm). |
| 2 | Throughput / rate limits (~10 seq calls) | gen | ✅ **done** | no throttling, 10/10 on both providers |
| 3 | End-to-end latency vs < 8 s ceiling | gen | ✅ **done** | embed 12 ms + gen ~4 s ≪ 8 s |
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

## 3. Latency — ✅ (embed + generate, both halves)

- **Local embed** (`all-MiniLM-L6-v2`): **dim = 384 confirmed** · cold load 11.6 s (one-time at startup) · **single-query embed = 12 ms** · batch ingest 377 chunks/s.
- **Generation** (probe via OpenRouter/OpenAI, 2026-06-08): single full-answer call ~3.75–4.40 s; warm short calls 0.48–0.67 s. **embed 12 ms + gen ~4 s ≪ 8 s ceiling**, with headroom for retrieval + grader.

## 1–2. Generation provider probe — ✅ (real data on an open model)

Ran `spike/gen_probe.py` against two OpenAI-compatible providers (Nebius credit not yet in; OpenRouter
is the representative stand-in — it serves the **open** models Nebius will).

| Provider | Model | JSON mode | Single gen | Warm mean (10 seq) | Throttling |
|---|---|---|---|---|---|
| **OpenRouter** | `meta-llama/llama-3.1-70b-instruct` (OPEN) | ✅ **YES** | 3.75 s | 0.48 s | none (10/10) |
| OpenAI | `gpt-4o-mini` | ✅ YES | 4.40 s | 0.67 s | none (10/10) |

- **JSON mode works on an open Llama model** — the grounding probe returned clean
  `{"grounded": false, "reason": ...}`. → The refusal grader can use the **JSON-mode primary path**
  (design §7), not just the cosine-threshold fallback. Nebius serves the same model class, so this is
  expected to hold; **re-point + re-run when credit lands** (low-risk confirmation, not a gate).
- **No rate-limit throttling** on 10 sequential calls either provider.
- **Two working providers behind one OpenAI-compatible `ModelProvider.generate()` seam** = the graded
  "swap models/providers" demo is real today. Generation provider presets: **Nebius (mandatory final
  call)**, OpenRouter, OpenAI, local Gemma — all the same seam (base_url + key + model id).

## Path note (not eval-blocking)

The design docs reference `../CuratedRAGMaterials/` (i.e. inside `Week2-RAG_ContextEngineering/`). The folder actually lives at **`GenAcademy/CuratedRAGMaterials/`** — one level higher (`../../CuratedRAGMaterials/` from `genacademy-rag/`). Production-corpus loaders must use the correct path; update the design's path reference during planning.

## Spike verdict — ✅ COMPLETE, cleared to plan Phase 0

All Phase-0-relevant items confirmed with real data: GitHub pin clean, guidebook parses (no OCR),
embed dim=384 / 12 ms, generation JSON-mode + latency + rate-limits validated on an **open** model.
Pinecone deferred to Phase 2 (P0 uses Chroma).

**Only follow-up (non-blocking):** when the Nebius key arrives, set `NEBIUS_API_KEY` in `spike/.env`
and re-run `spike/gen_probe.py` to confirm the open-model JSON-mode + latency hold on Nebius
specifically — a one-line config re-point, not an architecture change. Run `spike/pinecone_probe.py`
only when Phase 2 starts.
