# CLAUDE.md — mirror of AGENTS.md

This project's rules live in **[`AGENTS.md`](AGENTS.md)** (tool-neutral source of truth). Read it.
Everything there applies to Claude Code with no exceptions.

Quick pointers:
- **Constitution:** `specs/mission.md` · `specs/tech-stack.md` · `specs/roadmap.md`
- **Design (under external review):** `docs/design.md`
- **Decision reasoning:** `docs/architecture-decisions.md`

Non-negotiables (full list in `AGENTS.md`):
1. No code until the implementation plan is approved (we're pre-plan: design under review).
2. Builder ≠ reviewer — a different model / fresh context reviews every non-trivial change.
3. Evidence before "done" — show `ruff` + `pytest` output, not "it should work".
4. Pure core / thin view; citations captured at ingest; refusal path is load-bearing.
5. Don't replicate the starter/reference solutions; don't couple to `legal-rag-private`.
