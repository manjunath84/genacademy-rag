# GenAcademy RAG — Phase 1 (Product Layer) Design

**Date:** 2026-06-09
**Status:** approved (brainstorming) → **reworked after Codex design review** (2026-06-09) → pending implementation plan
**Builds on:** Phase 0 (merged, PR #1 — green cited Q&A core + 15-question eval report)
**Source of scope:** `specs/roadmap.md` § Phase 1, `docs/design.md` §5–6
**Companion:** `docs/phase1-decisions-and-tradeoffs.md` (the options/tradeoff rationale behind every decision here)

---

## 1. Goal

Turn the green Phase-0 spine (member asks → cited answer or refusal) into a **multi-user knowledge product**: real signup behind an invite gate, admin-managed content, and an admin usage dashboard. Everything stays **pure core / thin view**, the **`eval` collection stays immutable**, and the **refusal path stays load-bearing**.

**Guiding philosophy: "prod-grade but tight."** Pay for what's expensive to change later (schema, security primitives, architectural boundaries, concurrency model); defer what bolts on cheaply later (more loaders, repo-HEAD tracking, the datastore split, charting).

**Demoable skeleton:** admin logs in → generates an invite → new member redeems it → admin uploads a doc and watches it become queryable → admin deletes it → admin sees usage stats; member logs in and chats.

---

## 2. Scope

### MUST
1. **RBAC + invite-code signup** — admin-generated, single-use, role-bound, expiring, revocable invite codes; real `/signup`; passwords and invite-code secrets hashed with bcrypt.
2. **Admin content management** — list / upload / delete / re-index *uploaded* documents.
3. **`usage_log` + admin dashboard** — queries over time, top questions, refusal rate, latency p50/p95, grader-fallback rate.

### Cross-cutting prod-grade requirements (added in review)
- **CSRF protection** on every state-changing POST (login/logout if added, signup, `/ask` once it writes usage, invite generate/revoke, doc delete/reindex, upload).
- **Concurrency-safe corpus mutation** — a single retrieval snapshot swapped atomically + a mutation lock; no torn reads during upload/delete/reindex.
- **Idempotent schema + data migration** — add new columns/tables and rehash seeded plaintext passwords on an existing DB.

### SHOULD (defer unless cheap)
- Additional loaders (web page, PPTX, Python, JSON) into the existing registry.
- Production corpus tracking repo HEAD with a re-index trigger.

### Out of scope (explicit non-goals)
- OAuth / external identity providers (risk cap: session-based auth only).
- `Datastore` → `UserStore`/`DocStore`/`UsageStore` split (Postgres-era; documented extension point).
- Multi-worker / horizontally-scaled index (the snapshot model is correct for a single uvicorn worker with a threadpool; a shared/external index is a Phase-2 deploy concern).
- Charting libraries (inline SVG only).

---

## 3. Architecture (pure core / thin view — boundary unchanged)

```
core/                              data/                          web/ (thin view)
  security.py        (NEW, pure)     datastore.py  (GROWS)          app.py (GROWS)
    hash_password/verify             + _migrate() (idempotent)        /signup GET/POST
    new_invite_code() -> (id,        + invite_codes, usage_log         /admin/invites GET/POST/revoke
        secret, code_hash)             tables; documents +columns       /admin/documents GET + delete/reindex
    verify_invite_secret             + create_user                      /admin/dashboard GET
  analytics.py       (NEW, pure)     + redeem_invite() (atomic txn)     /ask (+ usage logging at the view)
    usage_summary(rows)->dict        + generate/list/revoke invite     require_admin guard
  retriever.py       (REFACTOR)      + delete_document (tombstone)      csrf token helper (view-only)
    immutable _Index snapshot        + log_query / recent_usage
    + mutation lock
  vectorstore.py     (GROWS)
    + delete_doc(doc_id)
```

**Invariants preserved from Phase 0:**
- No `fastapi`/template imports in `core/` or `data/`. The two new `core/` modules (`security`, `analytics`) and the CSRF helper that imports nothing from core are pure/view-local and offline-testable.
- **Usage logging lives in the view** — the `/ask` route times the call and writes `usage_log`; `QueryPipeline` imports no datastore. Observability at the IO boundary.
- **Mutations only touch `serving`.** `eval` is read-only; a test asserts product routes never instantiate or mutate the `eval` collection.
- **Refusal path untouched** — no new path emits an answer; Phase 1 wraps the existing graph with auth/content/observability.

**Datastore stays a single `SQLiteDatastore`** (YAGNI on the split), methods grouped by concern.

---

## 4. Security primitives (`core/security.py`, pure)

### 4.1 Passwords
bcrypt (`bcrypt` pinned dep). `_migrate()`/`seed_users()` detect any stored password that is not a bcrypt hash (Phase-0 plaintext) and rehash it. Login verifies against the hash.

### 4.2 Invite codes — structured token, not a bare hash *(review fix #1)*
A salted bcrypt hash **cannot be a lookup key** (every hash of the same input differs). So an invite code is a **structured bearer token**, like a Stripe/GitHub key:

```
code shown once  =  "<id>.<secret>"
  id      = secrets.token_urlsafe(8)   -> stored in clear (non-secret lookup handle)
  secret  = secrets.token_urlsafe(24)  -> NEVER stored; only bcrypt(secret) is stored
```

- **Generate:** `new_invite_code()` returns `(id, secret, secret_hash)`; the view shows `f"{id}.{secret}"` exactly once with a copy button; the datastore stores `id` (PK) + `secret_hash`.
- **Redeem:** split on the last `.`; look up the row by `id` (O(1)); bcrypt-verify `secret`; check lifecycle. Any mismatch → one generic error (no enumeration leak).

This gives O(1) lookup, hash-at-rest, show-once, and revocation/expiry — without scanning every bcrypt hash.

---

## 5. Data model

### New / changed tables
```sql
CREATE TABLE invite_codes (
  id          TEXT PRIMARY KEY,        -- public lookup handle (non-secret), first half of the code
  secret_hash TEXT NOT NULL,           -- bcrypt(secret half); raw secret never stored
  role        TEXT NOT NULL CHECK(role IN ('admin','member')),
  created_by  TEXT NOT NULL,           -- admin email
  created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
  expires_at  TEXT,                    -- NULL = never (UI default: +7 days)
  used_by     TEXT, used_at TEXT,      -- single-use audit (NULL = unused)
  revoked_at  TEXT                     -- admin kill switch (NULL = active)
);

CREATE TABLE usage_log (
  id INTEGER PRIMARY KEY, ts TEXT DEFAULT CURRENT_TIMESTAMP,
  user_email TEXT, question TEXT,
  refused INTEGER, confidence INTEGER, used_fallback INTEGER,   -- grader-health signal
  n_citations INTEGER, latency_ms INTEGER
);
```
`documents` gains (via idempotent `ALTER TABLE`): `deleted_at TEXT`, `deleted_by TEXT` *(review fix #5)*, `stored_path TEXT` *(review fix #8)*. `users.password` now holds a bcrypt hash.

**Redemption validity:** `used_at IS NULL AND revoked_at IS NULL AND (expires_at IS NULL OR expires_at > now)` AND `bcrypt_verify(secret, secret_hash)`.

### Schema + data migration *(review fix #6)*
`SQLiteDatastore` runs an idempotent `_migrate()` on init:
- `CREATE TABLE IF NOT EXISTS` for `invite_codes`, `usage_log`.
- For `documents`: `PRAGMA table_info(documents)` → `ALTER TABLE … ADD COLUMN` for any of `deleted_at`/`deleted_by`/`stored_path` not present.
- Seed/migrate users: rehash any non-bcrypt password to bcrypt.
- Safe to run on every boot (the dev `data/genacademy.sqlite` from Phase 0 upgrades in place).

---

## 6. Feature behavior

### 6.1 RBAC + signup
- `/signup` (GET form, POST): email + password + code → `Datastore.redeem_invite(...)` (see §6.4) → on success, session login with the code's role. Any failure → generic "invalid or expired code."
- `/admin/invites`: list issued codes with derived status (active/used/expired/revoked); POST generate (role + expiry; code shown once); POST revoke (`revoked_at=now`).
- `require_admin`: reusable view guard for all `/admin/*` routes (formalizes the Phase-0 `session.role != 'admin'` check). Optionally re-reads the role from the DB so a revoked/downgraded user can't ride a stale cookie *(review suggestion #9)*.

### 6.2 Content management
- `/admin/documents` (GET): table (title, source_type, status, n_chunks, uploaded_by, created_at) + actions.
- **Upload** (exists): read bytes, compute the content-hash `doc_id`, store bytes at `uploads_dir/<safe_doc_id>.<ext>` where the on-disk name derives from that `doc_id` (not the user filename) → no collisions; record `stored_path`; keep original `filename` for display only *(review fix #8)*. Parsing/chunking/embedding can happen before the corpus lock. The final persistence step runs under the corpus lock: write document/chunk metadata with the DB lock nested inside the corpus lock, upsert chunks into `serving`, then atomically swap the retrieval snapshot. `uploaded_by` is threaded from the session through the loader and `IngestPipeline.add_document(...)` *(review fix #3)*.
- **Delete** (POST, guarded `uploaded_by IS NOT NULL` — curriculum undeletable) — **precise, fail-safe, idempotent order** *(review fixes #2, #5)*:
  1. Acquire the corpus mutation lock.
  2. `VectorStore.delete_doc(doc_id)` on `serving` *(review fix #4)*, then rebuild + **atomically swap** the retrieval snapshot (§6.5) → doc is immediately un-queryable. (Correctness-critical step first.)
  3. `Path(stored_path).unlink(missing_ok=True)` (idempotent).
  4. Drop `chunks_meta` rows for the doc.
  5. Tombstone last: `UPDATE documents SET status='deleted', deleted_at=now, deleted_by=admin`.
  Re-runnable: a crash after step 2 leaves a non-queryable doc whose delete can simply be re-invoked. The `eval` collection is never touched.
- **Re-index** (POST): rebuild the snapshot from current `serving` chunks under the lock. Repo-HEAD re-pull deferred.

### 6.3 Usage dashboard
- `/ask`: time `QueryPipeline.answer`, write one `usage_log` row (question, refused, confidence, used_fallback, n_citations, latency_ms, user_email). Core untouched.
- `/admin/dashboard` (GET): `Datastore.recent_usage()` → rows → pure `usage_summary(rows)` → total, queries/day series, top-N questions (exact grouping), refusal rate, **latency p50/p95** (computed in Python), fallback rate → server-rendered cards + table + inline-SVG bar.

### 6.4 Atomic invite redemption *(review fix #1)*
`Datastore.redeem_invite(*, raw_code, email, password_hash) -> dict | None` runs **one transaction** (`BEGIN IMMEDIATE` to take the write lock up front):
1. Split `raw_code` → `id`, `secret`; `SELECT … WHERE id=?`.
2. Verify lifecycle + `bcrypt_verify(secret, secret_hash)`; on any failure → rollback, return `None`.
3. Conditional consume: `UPDATE invite_codes SET used_by=?, used_at=now WHERE id=? AND used_at IS NULL AND revoked_at IS NULL AND (expires_at IS NULL OR expires_at>now)`; assert exactly one row changed (closes the TOCTOU window).
4. `INSERT INTO users(...)` with the code's role in the **same** transaction; commit.
Two concurrent redemptions of the same code → exactly one wins (the conditional `UPDATE` rowcount), the other rolls back. A test asserts this.

### 6.5 Concurrency model *(review fix #2; revised after the 2nd review pass)*
`/ask` reads the retriever's index while `upload`/`delete`/`reindex` rewrites it, and FastAPI runs sync routes in a **threadpool** — real parallelism even on one worker. Two distinct hazards:
1. **Torn reads** inside the in-memory index — Phase-0 `reindex()` mutates three fields in sequence.
2. **Cross-store coherence** — `retrieve()` takes dense hits from *live Chroma* (`store.query`) and sparse hits from the *in-memory* index. These are **two consistency domains**; a lock-free snapshot alone can't unify them. A query holding the pre-delete snapshot can still rank a just-deleted chunk via its old BM25 and materialize it from its old `chunks_by_id` — the orphan window the 2nd review flagged. (So "materialize only ids in the bound snapshot" is *not* sufficient and is dropped as the mechanism.)

**Design:**
- `HybridRetriever`'s three fields become one immutable `_Index(ids, chunks_by_id, bm25)`; `reindex()` builds a new `_Index` and swaps the reference in a single assignment (no torn in-memory state).
- **A single `threading.Lock` (the corpus lock) guards both `retrieve()`'s index access AND every mutation** (Chroma `delete_doc`/`upsert` + snapshot swap). This is the correctness mechanism: the retrieval phase runs entirely before or entirely after a mutation, so dense (Chroma) and sparse (snapshot) are always the same corpus version — the orphan window is closed.
- Implementation detail: keep one shared corpus-lock object for the serving retriever and product mutation closures. Do not expose a pattern where route code acquires a non-reentrant lock and then calls a public `reindex()` that tries to acquire it again. Either expose a single `HybridRetriever.mutate_corpus(fn)`/`reindex_from_store(fn)` method that locks once and swaps an `_Index`, or keep a private `_swap_index_unlocked()` used by locked mutation methods. Product code never receives or mutates an `eval` store.
- The lock is held only for the retrieval node's index access (query embed + Chroma query + BM25 scoring + materialization); the seconds-long grade/answer **LLM calls run in later graph nodes, outside the lock**, so readers serializing with each other is negligible against LLM-dominated latency. An in-flight request that already completed retrieval before an admin delete may still finish with those pre-delete citations; after the delete's corpus mutation begins, no later retrieval can return the deleted chunks.
- *Escalation:* under multiple workers an in-process lock no longer suffices → external/shared index or a distributed lock (Phase-2 deploy). A read/write lock is the optimization if query concurrency matters before then; a plain mutex is the tight, correct Phase-1 choice.

### 6.6 CSRF *(review fix #7)*
A view-layer CSRF helper: a per-session token stored in the session, emitted as a hidden field in every state-changing form, validated on each POST (login/logout if added, signup, `/ask` once it writes `usage_log`, invite generate/revoke, doc delete/reindex, upload). Pure-core boundary untouched (the helper lives in `web/`). Session cookie set `same_site='lax'`; `secure=True` under HTTPS at deploy.

### 6.7 Datastore concurrency *(2nd review pass)*
The Phase-0 `SQLiteDatastore` holds **one** connection opened with `check_same_thread=False`. Phase 1 adds concurrent route writes (`redeem_invite`, `log_query`, invite revoke, delete tombstone, chunks cleanup) issued from threadpool threads sharing that single connection — unsafe (interleaved statements, and `BEGIN IMMEDIATE` on a shared connection is fragile).

**Design:** a datastore-level `threading.RLock` (the **DB lock**) wraps every connection use — all reads, writes, and transactions, including the `BEGIN IMMEDIATE` redemption transaction (§6.4).

**Lock ordering (deadlock avoidance):** the **corpus lock is always acquired before the DB lock**; no path holds the DB lock and then waits for the corpus lock. Route guards may re-read the user role from the DB, but they must release the DB lock before route code enters any corpus mutation.

Route matrix:
- `/ask`: `retrieve()` briefly takes the corpus lock, releases it before `grade`/`answer`, then the view writes `usage_log` under the DB lock. It never holds both locks at once.
- `upload`: read file/hash/parse/embed before locks where possible; final persistence takes corpus lock, then DB lock for document/chunk metadata, then `serving` upsert + snapshot swap while still under corpus lock.
- `delete`: corpus lock for `serving.delete_doc(...)` + snapshot swap, then DB lock for chunk cleanup + tombstone.
- `reindex`: corpus lock only for `serving.get_all_chunks()` + snapshot swap; any admin-list DB reads happen before or after, not while waiting on the corpus lock.
- `redeem_invite` / `log_query` / invite revoke / dashboard reads / login and signup user reads take only the DB lock.

(Alternative considered: short-lived per-operation connections + `busy_timeout` — more "real" for multi-process, more churn; the single-connection + RLock is the tight single-worker choice, with connection pooling as the Postgres-era path.)

---

## 7. Testing strategy

- **Pure modules offline:** `security` (hash≠plaintext, verify round-trip, wrong secret fails, code id/secret split, expiry/used/revoked logic); `usage_summary` (p50/p95, top-N, rates on canned rows, empty input).
- **Datastore (sqlite tmpfile):** `_migrate` adds columns to a Phase-0-shaped DB and rehashes plaintext; create_user; generate→redeem→used; redeem of expired/used/revoked/bad-secret rejected; **concurrent redeem → exactly one success**; delete_document tombstones (+deleted_by) and removes chunks; log_query/recent_usage round-trip.
- **Retriever:** snapshot swap leaves no torn read; with the corpus lock held around `retrieve()`, a delete cannot interleave a query, so a deleted chunk is never returned (orphan-window closed across the Chroma+BM25 boundary); reindex rebuilds BM25. A test exercises a delete concurrent with retrievals and asserts no deleted chunk surfaces.
- **Datastore concurrency:** concurrent `log_query` / `redeem_invite` from multiple threads don't corrupt the shared connection (DB-lock serialization); the redeem transaction stays atomic under contention.
- **Routes (TestClient, offline):** signup happy + each sad path; `require_admin` blocks members from every `/admin/*`; **CSRF token required** on each state-changing POST, including `/ask` after it writes usage (missing/wrong token → rejected); delete removes from `serving` but **`eval` stays pristine** (extends the Phase-0 two-collection assertion); upload stores under a doc-id path (no filename collision); `/ask` writes exactly one usage row; dashboard renders from seeded usage.
- All offline-deterministic via the existing `FakeModelProvider`; the single live `@pytest.mark.integration` test is unchanged.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| TOCTOU double-redeem of an invite | Single transaction + conditional `UPDATE … WHERE used_at IS NULL`, assert rowcount==1 (§6.4) |
| Salted hash isn't lookupable | Structured `id.secret` token: clear `id` for lookup, bcrypt(secret) at rest (§4.2) |
| Torn read / queryable orphan during delete/reindex | Immutable snapshot + atomic swap, and the **corpus lock around `retrieve()` itself** so dense (Chroma) + sparse (snapshot) stay one coherent version (§6.5) |
| Shared SQLite connection corrupted by concurrent threadpool writes | Datastore `threading.RLock` around all connection use; documented corpus→DB lock ordering (§6.7) |
| Stores desync on partial delete | Fail-safe order (index first), idempotent re-runnable delete, `unlink(missing_ok=True)` (§6.2) |
| CSRF on authenticated destructive POSTs | Per-session CSRF token on every state-changing form (§6.6) |
| Existing dev DB lacks new columns / has plaintext seeds | Idempotent `_migrate()` (PRAGMA + ALTER) + password rehash (§5) |
| Upload filename collision | On-disk name from content-hash doc_id; `stored_path` column; original filename display-only (§6.2) |
| Admin nukes the curriculum | Delete guarded to `uploaded_by IS NOT NULL`; `eval` never touched |
| In-memory snapshot wrong under multiple workers | Single-worker assumption documented; shared/external index = Phase-2 deploy |
| Question text in `usage_log` is PII-ish | Acceptable for a closed cohort tool; retention/redaction noted as an extension |

---

## 9. Review resolutions (Codex design review, 2026-06-09)

Verdict was **NEEDS-REWORK**; all 8 findings verified against the actual Phase-0 code and **accepted** (no rejections), folded in above:
1. **Invite redemption (Blocking):** atomic `redeem_invite()` transaction (§6.4) + structured `id.secret` token so a salted hash is lookupable (§4.2).
2. **Delete/reindex concurrency (Blocking):** immutable retrieval snapshot + atomic swap + mutation lock; refactor `HybridRetriever` (§6.5).
3. **`uploaded_by` (Important):** add to `Document`, thread loader → `IngestPipeline.add_document` so the delete guard works.
4. **VectorStore delete (Important):** add `delete_doc(doc_id)` to the protocol + `ChromaStore`.
5. **Tombstone (Important):** add `deleted_by`; precise idempotent delete order (§6.2).
6. **Migration (Important):** idempotent `_migrate()` — PRAGMA + ALTER + password rehash (§5).
7. **CSRF (Important):** view-layer per-session token on state-changing POSTs (§6.6).
8. **Upload collision (Important):** content-hash-derived on-disk path + `stored_path` (§6.2).
9. **Cookie hardening (Suggestion):** SameSite=lax now, Secure at deploy, optional DB role re-check in `require_admin` (§6.1).

### Second review pass (Codex, 2026-06-09) — 7/8 confirmed resolved; 2 new items, both folded in:
- **Blocking — cross-store orphan window:** the snapshot + "materialize-only-visible-ids" rule did *not* unify live-Chroma dense + in-memory sparse. Fixed by holding the **corpus lock around `retrieve()`** (not just mutations), so a query and a mutation never interleave (§6.5). The "materialize-only" rule is dropped as the mechanism.
- **Important — shared SQLite connection:** one `check_same_thread=False` connection under concurrent threadpool writes. Fixed with a datastore `threading.RLock` + documented corpus→DB lock ordering (§6.7).
- Regression checks passed: eval path stays read-only/reproducible (snapshot built from `eval_store.get_all_chunks()`); no new core/data boundary violation from CSRF or logging.

---

## 10. Net scope

2 new pure `core/` modules (`security`, `analytics`) · `HybridRetriever` snapshot refactor + **corpus lock around `retrieve()`** · `VectorStore.delete_doc` · datastore **`RLock`** (corpus→DB lock ordering) · 2 new tables + 3 `documents` columns + idempotent migration · ~6 routes + `require_admin` + CSRF helper · `uploaded_by` threaded through the ingest path · 1 new pinned dep (`bcrypt`). Larger than the first draft, but every addition closes a verified prod-grade gap.
