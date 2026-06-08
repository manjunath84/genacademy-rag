# AGENTS.md — Working Agreement (GenAcademy RAG)

**Applies to ALL agents — Claude Code, Codex, Cursor, the review agent, any future tool.** This is the
tool-neutral source of truth. Per-tool files (`CLAUDE.md`, etc.) are thin mirrors that point here.
Rules do not change with the tool.

*Status: draft, in force now. The design it governs (`docs/design.md`) is pending external review;
update this file if review changes the architecture.*

---

## 1. What we're building

A **knowledge assistant for Gen Academy cohort members** — ask anything about the cohort's curated
materials, get a **cited** answer or an honest **refusal**. Admins manage the corpus + monitor usage;
members chat. Fresh standalone build, **separate from `legal-rag-private`** (different product, no
privacy thesis). Full design: `docs/design.md`. Decision reasoning: `docs/architecture-decisions.md`.

## 2. The gates (no skipping)

1. **No code until the plan is approved.** Flow: brainstorm → `docs/design.md` (spec) → reviewed →
   implementation plan → build. Architecture is settled *before* implementation. We are currently
   **pre-plan** (design under review).
2. **Builder is never the sole judge.** The agent that writes code does **not** get the last word on
   whether it's correct or done. A **different model or a fresh context** reviews every non-trivial
   change. (Claude builds → Codex/other reviews, or vice-versa. Never one context grading itself.)
3. **Evidence before "done".** "It should work" is not done. Show **lint + test output** and, for
   user-facing behavior, a **live run / screenshot**. Green is something you *demonstrate*, not assert.
4. **Scope changes go through `specs/`.** Read `specs/roadmap.md` before expanding scope. Phase 0 ships
   and is *finished* before Phase 1 work begins.
5. **Eval green by Day 2.** The 15-question eval must be runnable and produce a scores table by end of
   Day 2. UI polish (streaming, source cards) is Day 3–4 *only if the eval is green*. Guards the #1
   failure mode: an impressive demo with a thin eval report. **"The eval" that may never be cut = the
   deterministic retrieval eval** (recall/precision/MRR + failure table) — that is the artifact the
   handout grades. The **LLM-as-judge faithfulness** layer is a depth add-on, **not** handout-required;
   it IS cuttable if the Nebius free tier throttles (§9 spike), falling back to the deterministic
   citation-grounding check. **Cut order if slipping:** SSE streaming → expandable source cards → admin
   upload UI → **LLM-judge faithfulness (→ citation-grounding fallback)** → **never** the retrieval eval
   or the refusal path.

## 3. Project guardrails (review-blockers — a reviewer should reject a PR that violates these)

- **Pure core / thin view.** All RAG + data logic lives in a testable core with **no** FastAPI/HTMX
  imports. Only the view layer touches HTTP/templates. A `from fastapi import` inside the RAG core is a
  reject.
- **Citations are captured at ingest, never reconstructed.** Every chunk carries
  `{doc_id, title, page/section, char_span}` from ingestion through to the answer. An answer that shows
  sources it didn't actually retrieve is a correctness bug.
- **The refusal path is load-bearing, not decorative.** When retrieval confidence is low, the system
  says "I could not find this in the course materials" — it does **not** answer from model priors. This
  is graded (the "unanswerable" eval question). Don't weaken it to make a demo look smarter.
- **Pluggability = interface + config, not branching.** Swap providers/stores via the `ModelProvider`
  / `VectorStore` / `Retriever` / `Datastore` interfaces + config presets. No `if provider == "nebius"`
  scattered through business logic.
- **The LLM never invents facts the corpus doesn't support, and never invents numbers.** Faithfulness
  to retrieved context is the product. (See sister skill `honest-ai-app`.)

## 4. Two cheap habits (mandatory)

- **Never quote a number/fact you haven't just re-derived.** Pricing, model param counts, context
  windows, dates *drift*. Re-check against the source before writing it down.
- **Reference calls are copied verbatim, never paraphrased.** Nebius/Pinecone/LangChain API
  signatures, **model IDs**, embedding **dimensions**, and request schemas are pasted from the official
  source into `specs/<feature>/requirements.md` — not reconstructed from memory. (The §9 spike in
  `docs/design.md` exists to capture these exactly.)

## 5. Hard "don'ts"

- **Do NOT replicate the starter/reference solutions.** The handout ships sample solutions and the repo
  contains an external reference (`../Knowledge-Intelligence-System/`). Per the handout, *replicating
  them scores zero.* Use them as hints at most; diverge deliberately and be able to explain why.
- **Do NOT couple this to `legal-rag-private`** or claim its privacy/on-prem thesis. This runs over
  non-sensitive course materials via a cloud API (Nebius). That claim would be false here.
- **Do NOT add product surface (auth, dashboards, extra ingest formats) ahead of a finished Phase 0.**
  Finishing the graded spine + eval is the priority. Scope creep is the main project risk.

## 6. Definition of done (per change)

- [ ] Lint clean (`ruff`) + tests pass (`pytest`) — **output shown**, not claimed.
- [ ] New behavior covered by a test (core logic) or a demonstrated run (UI).
- [ ] Reviewed by a different model / fresh context (gate #2).
- [ ] No guardrail (§3) violated.
- [ ] Docs/specs updated if scope or architecture moved.

## 7. Map of the project's own docs

- `specs/mission.md` — why · audience · in/out of scope.
- `specs/tech-stack.md` — layers + binding guardrails.
- `specs/roadmap.md` — phases, MUST vs SHOULD, risk caps.
- `docs/design.md` — full design (under external review).
- `docs/architecture-decisions.md` — locked stack decisions + reasoning.
- `docs/decisions/` — ADRs for significant choices made *during* the build.
