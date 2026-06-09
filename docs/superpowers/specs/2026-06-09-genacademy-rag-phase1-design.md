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
- **CSRF protection** on every state-changing POST (signup, invite generate/revoke, doc delete/reindex, upload).
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
- **Upload** (exists): store bytes at `uploads_dir/<safe_doc_id>.<ext>` where the on-disk name derives from the content-hash `doc_id` (not the user filename) → no collisions; record `stored_path`; keep original `filename` for display only *(review fix #8)*. `uploaded_by` is threaded from the session through the loader and `IngestPipeline.add_document(...)` *(review fix #3)*.
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

### 6.5 Concurrency model *(review fix #2)*
`HybridRetriever` is refactored so its three index fields become **one immutable snapshot object** `_Index(ids, chunks_by_id, bm25)`:
- `retrieve()` binds `snap = self._index` **once** at the top and reads only from `snap` for the whole call, and **materializes only ids present in `snap.chunks_by_id`** (so a vector deleted from Chroma under an in-flight query can't surface as an orphan).
- `reindex()` builds a brand-new `_Index` and does a single atomic assignment `self._index = new` (atomic under the GIL) — no torn intermediate state.
- A `threading.Lock` (the corpus mutation lock) serializes upload/delete/reindex so the Chroma delete and the snapshot swap happen together. `/ask` takes no write lock; it reads the current snapshot reference (atomic) → consistent snapshot isolation per query.

### 6.6 CSRF *(review fix #7)*
A view-layer CSRF helper: a per-session token stored in the session, emitted as a hidden field in every state-changing form, validated on each POST (signup, invite generate/revoke, doc delete/reindex, upload). Pure-core boundary untouched (the helper lives in `web/`). Session cookie set `same_site='lax'`; `secure=True` under HTTPS at deploy.

---

## 7. Testing strategy

- **Pure modules offline:** `security` (hash≠plaintext, verify round-trip, wrong secret fails, code id/secret split, expiry/used/revoked logic); `usage_summary` (p50/p95, top-N, rates on canned rows, empty input).
- **Datastore (sqlite tmpfile):** `_migrate` adds columns to a Phase-0-shaped DB and rehashes plaintext; create_user; generate→redeem→used; redeem of expired/used/revoked/bad-secret rejected; **concurrent redeem → exactly one success**; delete_document tombstones (+deleted_by) and removes chunks; log_query/recent_usage round-trip.
- **Retriever:** snapshot swap leaves no torn read; a chunk deleted from the store is not returned even by an in-flight query holding the old snapshot (orphan-window closed); reindex rebuilds BM25.
- **Routes (TestClient, offline):** signup happy + each sad path; `require_admin` blocks members from every `/admin/*`; **CSRF token required** on each state-changing POST (missing/wrong token → rejected); delete removes from `serving` but **`eval` stays pristine** (extends the Phase-0 two-collection assertion); upload stores under a doc-id path (no filename collision); `/ask` writes exactly one usage row; dashboard renders from seeded usage.
- All offline-deterministic via the existing `FakeModelProvider`; the single live `@pytest.mark.integration` test is unchanged.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| TOCTOU double-redeem of an invite | Single transaction + conditional `UPDATE … WHERE used_at IS NULL`, assert rowcount==1 (§6.4) |
| Salted hash isn't lookupable | Structured `id.secret` token: clear `id` for lookup, bcrypt(secret) at rest (§4.2) |
| Torn read / queryable orphan during delete/reindex | Immutable snapshot + atomic swap + per-call binding + materialize-only-visible-ids + mutation lock (§6.5) |
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

---

## 10. Net scope

2 new pure `core/` modules (`security`, `analytics`) · `HybridRetriever` snapshot refactor · `VectorStore.delete_doc` · 2 new tables + 3 `documents` columns + idempotent migration · ~6 routes + `require_admin` + CSRF helper · `uploaded_by` threaded through the ingest path · 1 new pinned dep (`bcrypt`). Larger than the first draft, but every addition closes a verified prod-grade gap.
