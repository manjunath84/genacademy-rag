# GenAcademy RAG — Phase 1 (Product Layer) Design

**Date:** 2026-06-09
**Status:** approved (brainstorming), pending implementation plan
**Builds on:** Phase 0 (merged, PR #1 — green cited Q&A core + 15-question eval report)
**Source of scope:** `specs/roadmap.md` § Phase 1, `docs/design.md` §5–6
**Companion:** `docs/phase1-decisions-and-tradeoffs.md` (the options/tradeoff rationale behind every decision here)

---

## 1. Goal

Turn the green Phase-0 spine (member asks → cited answer or refusal) into a **multi-user knowledge product**: real signup behind an invite gate, admin-managed content, and an admin usage dashboard. Everything stays **pure core / thin view**, the **`eval` collection stays immutable**, and the **refusal path stays load-bearing**.

**Guiding philosophy (set during brainstorming): "prod-grade but tight."** Pay for the things that are expensive to change later (schema, security primitives, architectural boundaries); defer the things that bolt on cheaply later (more loaders, repo-HEAD tracking, the datastore split, charting).

**Demoable skeleton at end of phase:** admin logs in → generates an invite → new member redeems it → admin uploads a doc and watches it become queryable → admin deletes it → admin sees usage stats; member logs in and chats.

---

## 2. Scope

### MUST
1. **RBAC + invite-code signup** — admin-generated, single-use, role-bound, expiring, revocable invite codes; real `/signup`; passwords and invite codes hashed (bcrypt).
2. **Admin content management** — list / upload / delete / re-index *uploaded* documents.
3. **`usage_log` + admin dashboard** — queries over time, top questions, refusal rate, latency p50/p95, grader-fallback rate.

### SHOULD (defer unless cheap)
- Additional loaders (web page, PPTX, Python, JSON) into the existing registry.
- Production corpus tracking repo HEAD with a re-index trigger.

### Out of scope (explicit non-goals)
- OAuth / external identity providers (risk cap: session-based auth only).
- The `Datastore` → `UserStore`/`DocStore`/`UsageStore` split (Postgres-era; documented extension point).
- Multi-worker / horizontally-scaled index (in-memory BM25 assumes a single uvicorn worker; a deploy concern → Phase 2).
- Charting libraries (inline SVG only).

---

## 3. Architecture (pure core / thin view — unchanged boundary)

```
core/                         data/                       web/ (thin view)
  security.py        (NEW)      datastore.py  (GROWS)        app.py        (GROWS: routes)
    hash_password()              + invite_codes table         /signup  GET/POST
    verify_password()            + usage_log   table          /admin/invites  GET/POST/revoke
    new_invite_code()            + signup / create_user       /admin/documents GET + delete/reindex
    hash_code() verify_code()    + invite CRUD + redeem        /admin/dashboard GET
  analytics.py       (NEW)       + log_query / recent_usage    /ask  (+ usage logging)
    usage_summary(rows)->dict    + delete_document (tombstone) auth/guard: require_admin
```

**Invariants preserved from Phase 0:**
- No `fastapi`/template imports in `core/` or `data/`. The two new `core/` modules (`security`, `analytics`) are **pure** (no IO) and fully offline-testable.
- **Usage logging lives in the view**, not in `QueryPipeline` — the route measures latency and writes `usage_log`; the core query path imports no datastore. Observability sits at the IO boundary.
- **Mutations (upload/delete/reindex) only touch the `serving` collection.** The `eval` collection is read-only and never mutated by the product layer.
- **Refusal path untouched** — Phase 1 adds no path that emits an answer; it only adds auth, content, and observability around the existing graph.

**Datastore stays a single `SQLiteDatastore`** (YAGNI on the split). Methods organized by concern (users / invites / documents / usage). Postgres + the three-store split is the documented Phase-2 extension.

**Concurrency:** upload/delete/reindex are serialized behind a single mutation lock; BM25 is rebuilt by building a new index and swapping the reference (atomic for a single worker). Multi-worker correctness is a Phase-2/deploy concern, noted not solved.

---

## 4. Security primitives (`core/security.py`, pure)

- **Passwords:** bcrypt (`bcrypt` pinned dep). `seed_users()` migrates the two Phase-0 users to hashed passwords. Login verifies against the hash.
- **Invite codes are bearer credentials → hashed at rest** with the same primitive. The raw code (high-entropy, `secrets.token_urlsafe`) is shown **once** at generation (copy-once UX); only `code_hash` is stored. Redemption verifies by hash.
- No password/code/secret is ever logged. (`GENACADEMY_SESSION_SECRET` already warns on the dev default.)

---

## 5. Data model (new tables)

```sql
CREATE TABLE invite_codes (
  code_hash   TEXT PRIMARY KEY,         -- bcrypt(raw code); raw never stored
  role        TEXT NOT NULL CHECK(role IN ('admin','member')),
  created_by  TEXT NOT NULL,            -- admin email
  created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
  expires_at  TEXT,                     -- NULL = never (UI default: +7 days)
  used_by     TEXT,                     -- redeemer email (NULL = unused)
  used_at     TEXT,
  revoked_at  TEXT                      -- admin kill switch (NULL = active)
);

CREATE TABLE usage_log (
  id           INTEGER PRIMARY KEY,
  ts           TEXT DEFAULT CURRENT_TIMESTAMP,
  user_email   TEXT,
  question     TEXT,
  refused      INTEGER,                 -- 0/1
  confidence   INTEGER,                 -- 1-5
  used_fallback INTEGER,                -- 0/1 — grader-degradation signal (PR-review fix)
  n_citations  INTEGER,
  latency_ms   INTEGER
);
```

`documents` gains `deleted_at TEXT` (tombstone). `users.password` now stores a bcrypt hash, not plaintext.

**An invite code is valid for redemption iff:** `used_at IS NULL AND revoked_at IS NULL AND (expires_at IS NULL OR expires_at > now)` AND the hash matches. Any failure → one generic "invalid or expired code" error (no enumeration leak).

---

## 6. Feature behavior

### 6.1 RBAC + signup
- `/signup` (GET form, POST redeem): email + password + invite code → validate code → create user with the code's role (bcrypt password) → mark code `used_by/used_at` → session login. Bad/expired/used/revoked code → generic error.
- `/admin/invites`: list issued codes with derived status (active / used / expired / revoked); POST generate (pick role + expiry, code shown once); POST revoke (set `revoked_at`).
- `require_admin` guard: a small reusable view dependency (formalizes the Phase-0 `session.role != 'admin'` check) protecting all `/admin/*` routes.

### 6.2 Content management
- `/admin/documents` (GET): table of documents (title, source_type, status, n_chunks, uploaded_by, created_at) with actions.
- **Delete** (POST, uploaded docs only): remove vectors from `serving` by `doc_id` → rebuild BM25 → drop `chunks_meta` → `documents.status='deleted', deleted_at=now` (tombstone) → remove the physical file. Guarded: `uploaded_by IS NOT NULL` (curriculum is undeletable).
- **Re-index** (POST): rebuild BM25 from current `serving` chunks (post-mutation consistency). Repo-HEAD re-pull is deferred.
- Upload (exists from Phase 0) moves under the documents view; filename already basename-sanitized (PR-review fix).

### 6.3 Usage dashboard
- `/ask` route: time the `QueryPipeline.answer` call, write a `usage_log` row (question, refused, confidence, used_fallback, n_citations, latency_ms, user_email). Core untouched.
- `/admin/dashboard` (GET): `Datastore.recent_usage()` returns rows → pure `usage_summary(rows)` computes: total queries, queries/day series, top-N questions (exact grouping), refusal rate, latency p50/p95, fallback rate → server-rendered cards + table + inline-SVG bar.

---

## 7. Testing strategy

- **Pure modules offline:** `security` (hash≠plaintext, verify round-trip, wrong password fails, code hash/verify, expiry/used/revoked logic), `usage_summary` (percentiles, top-N, rates on canned rows — no DB).
- **Datastore (sqlite tmpfile):** seed migrates to hashed; create_user; invite generate→redeem→used; redeem of expired/used/revoked rejected; delete_document tombstones + removes chunks; log_query/recent_usage round-trip.
- **Routes (TestClient, offline):** signup happy + each sad path; `require_admin` blocks members from every `/admin/*`; delete removes from `serving` but **`eval` stays pristine** (the Phase-0 two-collection assertion, extended); `/ask` writes exactly one usage row; dashboard renders from seeded usage.
- All offline-deterministic via the existing `FakeModelProvider` seam; no new live calls. Live provider stays the single `@pytest.mark.integration` test.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Delete leaves the three stores inconsistent on partial failure | Single mutation lock; delete order = index first (correctness), tombstone last; reindex idempotent and re-runnable |
| Admin nukes the seeded curriculum | Delete guarded to `uploaded_by IS NOT NULL`; `eval` collection never touched |
| In-memory BM25 wrong under multiple workers | Single-worker assumption documented; external/shared index is a Phase-2 deploy item |
| Question text in `usage_log` is PII-ish | Acceptable for a closed cohort tool; retention/redaction noted as an extension |
| Scope creep (more loaders, repo-HEAD, charts) | All explicitly SHOULD/deferred; MUST is bounded to 2 pure modules, 2 tables, ~6 routes |

---

## 9. Net scope

2 new pure `core/` modules · 2 new tables + `documents.deleted_at` · ~6 routes + an `require_admin` guard · 1 new pinned dep (`bcrypt`). Bounded, but genuinely shippable as a product layer.
