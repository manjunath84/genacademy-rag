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

**The catch a reviewer caught (and the fix).** A salted bcrypt hash is **not lookupable** — every
hash of the same input differs, so you can't `SELECT … WHERE code_hash = bcrypt(input)`. My first
draft made `code_hash` the primary key, which is meaningless for redemption lookup. Fix: structure
the code like a real API token — `code = "<id>.<secret>"` where `id` (8 url-safe bytes) is stored in
the **clear** as the lookup handle and only `bcrypt(secret)` is stored. Redeem = split on `.`, look
up by `id` (O(1)), then bcrypt-verify the secret. This is exactly how Stripe (`sk_live_…`) and GitHub
PATs (`ghp_…`) are shaped: a non-secret locator + a secret verifier.

**How an interviewer probes it.**
- *"Why bcrypt over SHA-256?"* → SHA-256 is fast = brute-force friendly; bcrypt has a tunable work
  factor + per-hash salt, designed to be slow. Mention argon2id as the modern alternative.
- *"How do you look up a credential you stored as a salted hash?"* → you can't hash-and-match; you
  need a non-secret lookup handle (the `id` half / a key prefix) and then verify the secret half
  against its hash. Naming *why* a salted hash defeats lookup is the signal. (Alternative at tiny
  scale: scan all active codes and verify each — O(n) bcrypt, fine for a handful, wrong at scale.)
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

**Decision: hybrid.** Precise delete order: (1) remove vectors from `serving` + atomically swap the
retrieval snapshot → un-queryable for later retrievals; (2) `unlink(missing_ok=True)` the file; (3)
drop `chunks_meta`; (4) tombstone the `documents` row last (`status='deleted', deleted_at,
deleted_by`). Serialized behind the corpus mutation lock; idempotent and re-runnable so a partial
failure recovers by re-running. Guarded so only *uploaded* docs are deletable — the seeded curriculum
can't be nuked — and **only `serving` is ever mutated; `eval` is immutable** (the Phase-0 isolation
invariant). The tombstone records `deleted_by` so "deleted by X at T" is answerable without a separate
audit table. Upload uses the opposite fail-safe bias: compute file/chunks/embeddings first, then under
the corpus lock write metadata with the DB lock nested, upsert to `serving`, and swap the snapshot so
the document is not searchable until the product metadata exists.

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

## Decision 7 — Concurrency: an immutable retrieval snapshot, not in-place mutation

**Problem.** `/ask` reads the retriever's index while `upload`/`delete`/`reindex` rewrites it. FastAPI
runs sync routes in a **threadpool**, so this is real parallelism even on one worker. The Phase-0
`reindex()` mutates three fields (`_chunks_by_id`, `_ids`, `_bm25`) in sequence — a query interleaving
mid-rebuild sees a **torn index** (new BM25, old id list), and a delete that removes vectors *then*
rebuilds BM25 leaves a window where a query still returns the deleted chunk.

**Options.** (a) a lock-free immutable snapshot swapped by one atomic reference assignment · (b) a lock
around both reads and mutations · (c) leave it (single-user demo).

**The two-pass story (worth telling as-is in an interview).** My first instinct was (a): the snapshot
gives lock-free reads — `retrieve()` binds the current snapshot, writers build a new one and swap the
reference (atomic under the GIL), so only writers contend. A second-model review **disproved that it's
sufficient**: `retrieve()` reads dense hits from *live Chroma* but sparse hits from the *in-memory*
snapshot — **two consistency domains**. A query holding the pre-delete snapshot can still rank a
just-deleted chunk via its old BM25 and return it; the GIL only protects the reference swap, not the
coherence of Chroma-plus-snapshot. So lock-free reads can't be made correct here without unifying the
two stores.

**Tradeoffs.** Unifying the stores (snapshot the dense vectors too) duplicates Chroma's job. Leaving it
ships the orphan bug. A lock around reads *and* writes makes a query and a mutation never interleave —
the simple correct answer — at the cost of serializing reads. But that cost is **negligible here**:
the lock only covers the ~12 ms of index access (embed + Chroma query + BM25); the seconds-long
grade/answer **LLM calls happen in later graph nodes, outside the lock**. Retrieval is not the
latency bottleneck, so serializing it costs ~nothing.

**Decision: immutable snapshot (clean atomic rebuild) + a single `threading.Lock` (the corpus lock)
held around `retrieve()` itself and every mutation.** The lock — not the snapshot — is the correctness
mechanism that keeps dense + sparse coherent per retrieval. It protects the graph's retrieve node, not
the later grade/answer nodes, so model calls never run under the corpus lock. A request that retrieved
before a delete may still finish with pre-delete citations; after the delete mutation starts, later
retrievals cannot see the deleted chunks. A read/write lock is the optimization if query concurrency
ever matters; a plain mutex is the tight Phase-1 choice. Multi-worker → external/shared lock
(Phase-2).

Implementation detail: use one shared corpus-lock object for the serving retriever and product
mutation closures. Avoid external code taking a non-reentrant lock and then calling a public
`reindex()` that takes it again; either route mutations through a single `mutate_corpus(fn)` /
`reindex_from_store(fn)` method that locks once, or use a private unlocked swap helper inside locked
methods. **Plus a datastore `threading.RLock`** around the shared SQLite connection (Phase-0 opened it
`check_same_thread=False`, and Phase 1 adds concurrent threadpool writes), with documented
**corpus-lock-before-DB-lock** ordering so the two locks can't deadlock. `/ask` takes corpus and DB
locks in separate phases (retrieve first, usage logging later); upload/delete take corpus then DB;
invite/login/dashboard paths take only DB. Admin role re-checks must release the DB lock before any
corpus mutation begins.

**How an interviewer probes it.**
- *"Single worker — is there even concurrency?"* → yes: sync routes run in a threadpool; async routes
  interleave at await points. "One process" ≠ "one thing at a time."
- *"A lock-free snapshot sounds elegant — why didn't it work?"* → because retrieval spans two stores
  (live Chroma + in-memory BM25); a snapshot makes the in-memory side coherent but not the pair. This
  is the trap: a snapshot gives you isolation *within one store*, not *across* stores.
- *"Doesn't locking reads kill throughput?"* → not when the locked section (~12 ms retrieval) is dwarfed
  by the unlocked LLM calls (seconds). Lock granularity matters more than lock/no-lock: hold it for the
  index access, never around the model call.
- *"Two locks — deadlock?"* → fixed acquisition order (corpus before DB), and no path takes them in the
  other order. Stating the ordering rule is the answer.
- *"Why is the reference swap atomic?"* → attribute assignment is atomic under the GIL; a reader sees
  the whole old or whole new snapshot — but (per above) that alone wasn't enough, which is the lesson.

---

## Decision 8 — CSRF on state-changing routes (the thing that's easy to forget)

**Problem.** Phase 1 adds authenticated destructive POSTs (delete doc, revoke invite, reindex, upload,
generate invite). Auth is a **session cookie**, which the browser sends automatically — including on a
request forged by another site. Role checks don't help: the victim *is* an admin.

**Options.** (a) nothing (rely on the role check — vulnerable) · (b) per-session CSRF token in a
hidden form field, validated on POST · (c) double-submit cookie · (d) SameSite cookie only.

**Tradeoffs.** Role checks authorize *who*, not *intent* — they don't stop CSRF. `SameSite=lax`
mitigates a lot but isn't a complete defense (and is a browser default, not a guarantee). The
synchronizer-token pattern (b) is the textbook server-rendered-form defense and is small + lives
entirely in the view, so it doesn't touch pure core.

**Decision: per-session CSRF token on every state-changing form + `SameSite=lax`.** Defense in depth,
view-layer only. This includes login/logout if present, signup, `/ask` after it writes `usage_log`,
invite generate/revoke, upload, delete, and reindex. Flagged as **Important** by the review; absent
from the first draft.

**How an interviewer probes it.**
- *"Why doesn't the admin role check stop CSRF?"* → CSRF rides the victim's own authenticated
  session; the forged request *is* authorized. You need to prove the request came from your own page.
- *"SameSite=lax — done?"* → helps, not sufficient (top-level GET navigations, older browsers, edge
  methods); keep the token as the real control.
- *"Where does the token live in a layered app?"* → the view (it's an HTTP/transport concern), never
  the domain core — same boundary discipline as usage logging.

---

## Decision 9 — Schema migration on an existing SQLite file (the unglamorous one)

**Problem.** Phase 0 created tables with `CREATE TABLE IF NOT EXISTS`. Phase 1 adds columns
(`documents.deleted_at/deleted_by/stored_path`) and must rehash plaintext seed passwords. On a fresh
DB that's automatic; on the **existing** `data/genacademy.sqlite`, `IF NOT EXISTS` is a no-op and the
new columns never appear — silent drift between code and schema.

**Options.** (a) drop/recreate the DB (loses data) · (b) a migration tool (Alembic — heavy for SQLite
+ one dev DB) · (c) a tiny idempotent in-code migration.

**Tradeoffs.** Drop-and-recreate is fine in dev but a habit that bites in prod. Alembic is the right
answer at scale but overkill here and adds a dep + workflow. A hand-rolled idempotent `_migrate()`
(`PRAGMA table_info` → `ALTER TABLE ADD COLUMN` for missing ones; rehash non-bcrypt passwords) is
~15 lines, runs safely on every boot, and teaches the actual mechanic.

**Decision: idempotent in-code `_migrate()` for Phase 1; note Alembic as the Postgres-era upgrade.**
Important finding from the review — the first draft said "migrate passwords" but never said *how*, and
ignored column migration entirely.

**How an interviewer probes it.**
- *"`CREATE TABLE IF NOT EXISTS` — what's the trap?"* → it never alters an existing table; new columns
  silently don't get added; you discover it as a runtime "no such column."
- *"How do you make a migration safe to run repeatedly?"* → check current state first (`PRAGMA
  table_info`) and only apply the delta; rehash only passwords that aren't already bcrypt.
- *"When do you reach for Alembic?"* → multiple environments, ordered/versioned migrations, rollbacks,
  a team — i.e., when "run a delta on boot" stops being safe or auditable.

---

## A note on process (a behavioral talking point in itself)

This design was reviewed by a **different model (Codex)** before any code — the project's "builder ≠
reviewer" rule — and it took **two passes**. Pass 1 returned **NEEDS-REWORK** with two Blocking
findings (the salted-hash lookup and the torn-index race) plus six more; all eight were verified
against the actual Phase-0 code and accepted. Pass 2 confirmed 7/8 fixed but caught that my
concurrency fix was *incomplete* — the snapshot didn't unify the live-Chroma + in-memory-BM25 stores,
leaving an orphan window — and flagged the shared SQLite connection. Both folded in (lock around
`retrieve()`; datastore `RLock`). The lesson worth repeating in an interview: *a second reviewer
catches the expensive class of bug — wrong data model, wrong concurrency model — while it's still a
paragraph to change; and the **re-review** matters, because the first fix to a concurrency bug is often
itself subtly incomplete.*

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
