# Phase 1 — Decisions & Tradeoffs (interview-prep knowledge base)

**Purpose:** a study artifact, not a spec. Every non-trivial Phase-1 fork written as
**Problem → Options → Tradeoffs → Decision → How an interviewer probes it.** The terse decision
record lives in `docs/architecture-decisions.md`; the spec lives in
`docs/superpowers/specs/2026-06-09-genacademy-rag-phase1-design.md`. This is the "why," at the depth
you'd want to defend in a system-design or behavioral interview.

**Overarching principle: "prod-grade but tight."** Pay upfront for what's *expensive to change later*
(schema shape, security primitives, architectural boundaries). Defer what *bolts on cheaply later*
(more loaders, repo-HEAD tracking, the datastore split, charts). This sentence is itself a strong
interview answer to "how do you decide what to build now vs. later?" — the axis is **cost-of-change**,
not effort.

---

## Decision 1 — Invite-code model (RBAC signup gate)

**Problem.** Phase 0 has two hardcoded seeded users. To become a real multi-user product we need
signup, but without OAuth (risk cap). How do new accounts get created and gated?

**Options.**
- **A. Admin-generated, single-use, role-bound codes.** Admin issues a code carrying a role; a new
  user redeems it once; the code is then consumed. New `invite_codes` table.
- **B. Shared reusable code per role.** Two long-lived secrets (member, admin) in config/db; anyone
  with the member code self-signs as a member.
- **C. Single static signup code.** One env code; anyone who knows it becomes a member; admins seeded.

**Tradeoffs.**
| | Revocable? | Auditable? | Leak blast radius | Build cost | "Prod-grade"? |
|---|---|---|---|---|---|
| A | ✅ per code | ✅ who/when issued+used | one invite | 1 table + redeem route | ✅ |
| B | ✅ but global (rotate the shared secret, breaks everyone) | ❌ can't tell who used what | whole role opens | tiny | ❌ |
| C | ❌ | ❌ | whole app opens | trivial | ❌ (toy) |

**Decision: A.** B and C are disqualified on *prod-grade*, not effort: a shared/static secret can't
be revoked or audited and a single leak opens signup to the world. A is the universal real-world
pattern (GitHub org invites, Slack, Stripe) and the cost delta is just one small table + a redeem
endpoint. Built with lifecycle columns from day one (`expires_at`, `used_by/used_at`, `revoked_at`)
so future extensions — max-uses, email-bound invites, teams — bolt on without a schema rewrite.

**How an interviewer probes it.**
- *"Why not a shared signup link?"* → revocation + audit + blast radius; a shared secret is a single
  point of total compromise with no forensic trail.
- *"Single-use — what about race conditions?"* → redemption must be atomic: check-and-mark `used_at`
  in one transaction (`UPDATE … WHERE used_at IS NULL` and assert one row changed), or two users
  could redeem the same code concurrently. Naming the race is the point.
- *"How do you extend to email-bound invites?"* → add an `email` column + check at redeem; the table
  shape already supports it, no migration of existing semantics.

---

## Decision 2 — Invite codes & passwords: hashing at rest

**Problem.** How are credentials stored? Phase 0 stored passwords in plaintext (explicitly a P0
shortcut). Invite codes are a new credential.

**Options.** (a) plaintext (P0 status quo) · (b) fast hash (SHA-256 + salt) · (c) adaptive hash
(bcrypt/argon2) · and for codes specifically: store raw vs store a hash + show-once.

**Tradeoffs.** Plaintext: a DB read = total account compromise. Fast hashes (SHA-256) are
brute-forceable at billions/sec on a GPU — wrong tool for secrets. **Adaptive hashes (bcrypt)** are
deliberately slow and salted, the industry default for passwords. An **invite code is a bearer
credential** (whoever holds it can create an account) so it deserves the same treatment as a
password: hash at rest, show the plaintext exactly once at generation (the GitHub PAT / API-key UX).

**Decision: bcrypt for both passwords and invite codes; codes shown once.** One security primitive,
two uses. `seed_users()` migrates the two P0 users to hashed. Cost is tiny because we build the util
once. "Invite codes are hashed like passwords because they're bearer credentials" is a deliberate
principal-level signal.

**How an interviewer probes it.**
- *"Why bcrypt over SHA-256?"* → SHA-256 is fast = brute-force friendly; bcrypt has a tunable work
  factor + per-hash salt, designed to be slow. Mention argon2id as the modern alternative.
- *"Why show the code only once?"* → if you can re-display it, you must store it recoverably
  (plaintext/encrypted) → defeats hashing. Show-once lets you store only the hash.
- *"Salt — where does it come from?"* → bcrypt embeds a random salt in the hash string; you don't
  manage it separately. Knowing this distinguishes "I used a library" from "I understand it."

---

## Decision 3 — Document delete: hard vs soft vs hybrid (the distributed-consistency one)

**Problem.** A document's state lives in **three stores**: the Chroma `serving` collection (vectors),
SQLite (`documents` + `chunks_meta`), and the uploads directory (the file). Delete must leave a
coherent system. This is a small distributed-consistency problem.

**Options.**
- **Hard delete:** physically remove from all three.
- **Soft delete:** mark `status='deleted'`, keep everything, filter it out at query time.
- **Hybrid (chosen):** hard-remove from the *search path* (vectors + BM25), keep a *tombstone* row
  (`status='deleted', deleted_at`) for audit, remove the file.

**Tradeoffs.**
- *Soft delete* is reversible and audit-friendly but forces **every retrieval and the BM25 rebuild to
  filter by status** — query-time complexity that's easy to get subtly wrong (BM25 is built from a
  chunk list; a stale "deleted" chunk silently re-enters results). It also keeps dead vectors in the
  index, degrading search.
- *Hard delete* is clean for search but **loses the audit trail** (who uploaded it, when, why it's
  gone) — and a mid-operation failure can desync the stores.
- *Hybrid* gives **index-reflects-reality immediately** (correctness) **and** keeps history. The
  relational tombstone is the system of record; the search index holds only live content.

**Decision: hybrid.** Delete order = **index first** (vectors + BM25 → immediately un-queryable),
**tombstone last**. Serialized behind a mutation lock; reindex is idempotent and re-runnable so a
partial failure is recoverable by re-running. Guarded so only *uploaded* docs are deletable — the
seeded curriculum can't be nuked — and **only the `serving` collection is ever mutated; `eval` is
immutable** (the Phase-0 isolation invariant, carried forward).

**How an interviewer probes it.**
- *"What if the process crashes mid-delete?"* → order matters: do the correctness-critical removal
  (search index) first; the tombstone is cosmetic-by-comparison; reindex is idempotent so re-running
  converges. Contrast with the naive order (file/row first) which can leave queryable orphans.
- *"Two stores, no distributed transaction — how do you stay consistent?"* → you can't get ACID
  across Chroma + SQLite + filesystem; you pick an ordering that fails *safe* (toward
  un-queryable, never toward serving deleted content) and make the repair operation idempotent.
- *"Why keep the tombstone?"* → audit/forensics + lets you show "deleted by X at T" without a
  separate audit log table.

---

## Decision 4 — Where does usage logging live? (layering / separation of concerns)

**Problem.** We need to record every query for the dashboard. Logging is IO (a DB write). The query
path is the `QueryPipeline` in the **pure core**, which has no datastore import by design.

**Options.** (a) log inside `QueryPipeline` · (b) log in the thin web view (the `/ask` route).

**Tradeoffs.** Logging in the core is "convenient" (one place) but **couples domain logic to a
datastore** and breaks the pure-core/thin-view boundary — now the core can't be unit-tested without a
DB, and a CLI/batch caller of the pipeline would silently write usage rows. Logging in the view keeps
the core a pure function (question → result) and puts the side-effect at the **IO boundary**, where
the request already lives and where latency is naturally measured.

**Decision: log in the view.** `QueryResult` already carries everything (answer, citations, refused,
confidence, `used_fallback`); the route adds the one thing only it knows — wall-clock latency — and
writes the row. Core stays pure and offline-testable.

**How an interviewer probes it.**
- *"Isn't that scattering logging across every route?"* → it's one route (`/ask`); if it grew, you'd
  extract a thin middleware/decorator at the *view* layer — still outside the core.
- *"Observability in a layered architecture — where does it go?"* → instrumentation belongs at
  boundaries (IO, request edges), not woven into domain logic; the domain emits *values* (we return
  `used_fallback` in the result), the edge decides what to *persist/emit*.
- This is the same principle we already applied in the PR-review fix: the grader *returns*
  `used_fallback`, the graph *logs* it, the route *persists* it — domain produces the signal, each
  boundary decides what to do with it.

---

## Decision 5 — Dashboard analytics: percentiles & "top questions" in SQLite

**Problem.** Dashboard needs latency p50/p95, refusal rate, queries/day, top questions. SQLite has no
`PERCENTILE` function.

**Options.** (a) approximate percentiles in SQL with window-function gymnastics · (b) pull recent rows
and compute in a pure Python helper · (c) add a heavier analytics store.

**Tradeoffs.** SQL percentile emulation is brittle and unreadable; a heavy store is absurd
over-engineering for a cohort tool. At this data scale, **pulling rows and computing in Python** is
trivial, exact, and — crucially — the computation becomes a **pure function** (`usage_summary(rows)
-> dict`) you can unit-test offline with canned rows, no DB.

**Decision: SQL does the cheap GROUP BYs (counts, rates, top-N grouping); a pure `usage_summary`
helper does percentiles + assembly.** Top questions use exact-string grouping (good enough;
normalization/clustering is a documented extension). "Top questions = `GROUP BY question`" is a
deliberate v1 — semantic clustering is a real feature, not a freebie.

**How an interviewer probes it.**
- *"What happens to 'compute in Python' at 10M rows?"* → it breaks; you'd push aggregation into the
  store (Postgres `percentile_cont`, or a rollup table / time-series DB). Knowing the scale ceiling of
  your own choice is the answer — it's right *for this scale*, and you can name where it stops being.
- *"Exact-string top questions — limitation?"* → 'reset password' vs 'how do I reset my password'
  count separately; fix is embedding-based clustering (which this very project already has the
  embedding stack for — a nice callback).
- *"p95 from a sample vs the population?"* → recent-N is a windowed/sampled percentile; fine for a
  dashboard trend, not for SLA billing.

---

## Decision 6 — Keep one `Datastore`, don't split yet (YAGNI vs. premature abstraction)

**Problem.** The datastore is growing (users, documents, chunks, now invites + usage). `design.md`
anticipates a `UserStore`/`DocStore`/`UsageStore` split. Do it now or later?

**Options.** (a) split now along concern boundaries · (b) keep one `SQLiteDatastore`, organize methods
by concern, split when Postgres arrives.

**Tradeoffs.** Splitting now adds interfaces + wiring for a benefit (independent backends, smaller
files) that **only pays off when a second backend or a real size problem exists** — neither does yet.
Premature splitting is speculative generality. But ignoring the seam entirely risks a god-object.
Middle path: one class, methods grouped by concern, the split *documented as the trigger-based
extension* (when the Postgres preset lands).

**Decision: keep one `SQLiteDatastore` for Phase 1; document the split as a Phase-2/Postgres trigger.**

**How an interviewer probes it.**
- *"When would you split it?"* → at a concrete trigger: a second storage backend, a file/ownership
  boundary that hurts, or independent scaling needs — not "it feels big." Trigger-based refactoring
  vs. speculative.
- *"Isn't this a god object?"* → it's bounded and grouped; the Protocol seam already exists so the
  split is mechanical when triggered. The cost of waiting is low *because* the seam is there.

---

## Cross-cutting themes (good behavioral/architecture talking points)

1. **Cost-of-change as the build-now/build-later axis** — schema, security, boundaries now; loaders,
   charts, repo-HEAD later.
2. **Fail-safe ordering over distributed transactions** — when you can't get ACID across stores
   (Chroma/SQLite/FS), choose an order that fails toward the safe state and make repair idempotent.
3. **Domain emits values; boundaries act on them** — `used_fallback` is returned by the grader,
   logged by the graph, persisted by the route, shown by the dashboard. One signal, four layers, no
   coupling.
4. **Invariants carried across phases** — `eval` immutable, refusal load-bearing, pure core / thin
   view. Phase 1 adds product surface *around* the core without reaching into it.
5. **Know the ceiling of your own choice** — "compute percentiles in Python" and "exact-string top
   questions" are right *for this scale*, and being able to say exactly where they stop scaling is
   stronger than pretending they're universal.
