# GenAcademy RAG Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 product layer: invite-gated RBAC signup, admin corpus management, usage logging/dashboard, and concurrency-safe serving-corpus mutation while preserving the immutable eval collection and refusal path.

**Architecture:** Keep the Phase-0 pure-core/thin-view split. Add pure `core/security.py` and `core/analytics.py`; grow `SQLiteDatastore` behind an `RLock`; refactor `HybridRetriever` to a locked immutable snapshot; keep all HTTP/session/CSRF/template work inside `web/`. Product mutations touch only the `serving` Chroma collection.

**Tech Stack:** Python 3.12, FastAPI/Jinja2/Starlette sessions, SQLite, ChromaDB, bcrypt, rank-bm25, LangGraph, pytest, ruff, uv.

---

## Phase A Verdict

APPROVE-WITH-REVISIONS. The final design review found two underspecified areas and fixed them before this plan:

- §6.5 now states the corpus lock protects the retrieval node only; grade/answer LLM calls run after `retrieve()` returns.
- §6.7 now includes a route lock matrix, including upload and admin role re-check behavior, so no path acquires DB lock then waits on corpus lock.
- CSRF coverage now includes `/ask` after it writes usage, plus login/logout if present.

## Guardrails

- Do not read or ingest sibling/reference projects. Stay inside `genacademy-rag`.
- No FastAPI, Starlette, Jinja2, or template imports in `src/genacademy_rag/core/` or `src/genacademy_rag/data/`.
- Only `serving` is mutated by web/product routes. `eval` remains read-only and `scripts/eval_retrieval.py` must keep the same metrics.
- The graph refusal branch is not bypassed. `/ask` wraps `QueryPipeline.answer()` and logs the returned `QueryResult`; it does not generate answers directly.
- Secrets are read from env/session/user input only. Invite secrets are shown once and stored only as bcrypt hashes.
- Commit after every task with this trailer:

```text
Co-Authored-By: Codex <noreply@openai.com>
```

Context7 docs consulted for library API shape:

- `/pyca/bcrypt`: `bcrypt.hashpw(password_bytes, bcrypt.gensalt())` and `bcrypt.checkpw(candidate_bytes, hash_bytes)`.
- `/websites/cookbook_chromadb_dev`: `collection.delete(where={...})` and metadata filters on `Collection.get()` / `Collection.query()`.
- `/fastapi/fastapi`: `Form`, `File`, `UploadFile`, `Request`, `TestClient` form posts, and template responses.

## File Structure

- Create `src/genacademy_rag/core/security.py`: pure bcrypt password/invite-secret helpers, structured invite token generation/parsing.
- Create `tests/core/test_security.py`: offline tests for hashing, verification, malformed invite codes, and long secrets.
- Modify `pyproject.toml`: add pinned `bcrypt`.
- Modify `src/genacademy_rag/data/datastore.py`: `RLock`, idempotent migration, invite lifecycle, user creation, document tombstone, usage logging.
- Expand `tests/data/test_datastore.py`: migration, password rehash, invite lifecycle, concurrent redeem, concurrent writes, documents, usage log.
- Modify `src/genacademy_rag/web/auth.py`: authenticate with bcrypt verification.
- Modify `src/genacademy_rag/core/vectorstore.py`: add `delete_doc(doc_id)` to the protocol and Chroma implementation.
- Expand `tests/core/test_vectorstore.py`: Chroma `delete_doc` removes only matching document chunks.
- Modify `src/genacademy_rag/core/types.py`: add `Document.uploaded_by` and `Document.stored_path`.
- Modify `src/genacademy_rag/core/loaders/pdf_loader.py`: preserve `uploaded_by` and optional `stored_path`.
- Modify `src/genacademy_rag/core/pipeline.py`: pass `uploaded_by` and `stored_path` into datastore; write metadata before vector upsert.
- Expand `tests/core/test_pdf_loader.py` and `tests/core/test_ingest_pipeline.py`: provenance is threaded.
- Modify `src/genacademy_rag/core/retriever.py`: immutable `_Index`, corpus lock, locked `retrieve()`, locked mutation/reindex helper.
- Expand `tests/core/test_retriever.py`: no torn read and no deleted orphan after concurrent mutation.
- Create `src/genacademy_rag/core/analytics.py`: pure `usage_summary(rows, top_n=5)`.
- Create `tests/core/test_analytics.py`: percentiles, rates, top questions, empty input.
- Modify `src/genacademy_rag/web/app.py`: CSRF helper, `require_admin`, signup, invite admin routes, documents admin routes, ask usage logging, dashboard.
- Add templates:
  - `src/genacademy_rag/web/templates/signup.html`
  - `src/genacademy_rag/web/templates/admin_invites.html`
  - `src/genacademy_rag/web/templates/admin_documents.html`
  - `src/genacademy_rag/web/templates/admin_dashboard.html`
- Modify templates:
  - `src/genacademy_rag/web/templates/login.html`
  - `src/genacademy_rag/web/templates/chat.html`
- Expand `tests/web/test_app.py`: signup, admin guard, CSRF, upload filename collision, delete serving-only, usage logging, dashboard rendering.

---

## Task 0: Baseline And Branch Check

**Files:** none

- [ ] **Step 1: Confirm branch and clean state**

Run:

```bash
git status --short --branch
git rev-parse --short HEAD
```

Expected: branch is `feat/genacademy-rag-phase1`; no uncommitted files except this plan if it has not been committed yet.

- [ ] **Step 2: Capture current deterministic eval output**

Run:

```bash
uv run python scripts/eval_retrieval.py
```

Expected: command exits 0 and prints one `RETRIEVAL EVAL` summary line plus per-question rows. Save the summary line in working notes for final comparison after implementation.

- [ ] **Step 3: Run current tests**

Run:

```bash
uv run pytest
```

Expected: all non-integration tests pass and the integration test is deselected by `pyproject.toml` addopts.

No commit for Task 0.

---

## Task 1: Pure Security Helpers

**Files:**
- Create: `src/genacademy_rag/core/security.py`
- Create: `tests/core/test_security.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add failing security tests**

Create `tests/core/test_security.py`:

```python
from genacademy_rag.core.security import (
    hash_password,
    hash_secret,
    is_bcrypt_hash,
    new_invite_code,
    split_invite_code,
    verify_password,
    verify_secret,
)


def test_password_hash_round_trip_and_wrong_password_fails():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert is_bcrypt_hash(hashed)
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong", hashed)


def test_long_password_is_supported_before_bcrypt_limit():
    long_password = "x" * 200
    hashed = hash_password(long_password)
    assert verify_password(long_password, hashed)
    assert not verify_password(long_password + "y", hashed)


def test_invite_code_is_structured_and_secret_hash_verifies():
    code_id, secret, secret_hash = new_invite_code()
    raw_code = f"{code_id}.{secret}"
    assert "." in raw_code
    assert secret not in secret_hash
    assert is_bcrypt_hash(secret_hash)
    assert split_invite_code(raw_code) == (code_id, secret)
    assert verify_secret(secret, secret_hash)
    assert not verify_secret("wrong-secret", secret_hash)


def test_malformed_invite_code_returns_none():
    assert split_invite_code("") is None
    assert split_invite_code("missing-dot") is None
    assert split_invite_code(".missing-id") is None
    assert split_invite_code("missing-secret.") is None
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
uv run pytest tests/core/test_security.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'genacademy_rag.core.security'`.

- [ ] **Step 3: Add bcrypt dependency**

Modify `pyproject.toml` dependencies:

```toml
    "bcrypt==5.0.0",
```

Run:

```bash
uv lock
```

Expected: exits 0 and updates `uv.lock` with `bcrypt==5.0.0`.

- [ ] **Step 4: Implement `core/security.py`**

Create `src/genacademy_rag/core/security.py`:

```python
"""Pure security helpers for passwords and invite-code bearer secrets."""
from __future__ import annotations

import base64
import hashlib
import secrets

import bcrypt

BCRYPT_PREFIXES = ("$2a$", "$2b$", "$2y$")


def _bcrypt_input(value: str) -> bytes:
    raw = value.encode("utf-8")
    if len(raw) <= 72:
        return raw
    return base64.b64encode(hashlib.sha256(raw).digest())


def is_bcrypt_hash(value: str | None) -> bool:
    return bool(value and value.startswith(BCRYPT_PREFIXES))


def hash_secret(secret: str) -> str:
    return bcrypt.hashpw(_bcrypt_input(secret), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_secret(secret: str, secret_hash: str) -> bool:
    if not is_bcrypt_hash(secret_hash):
        return False
    try:
        return bcrypt.checkpw(_bcrypt_input(secret), secret_hash.encode("utf-8"))
    except ValueError:
        return False


def hash_password(password: str) -> str:
    return hash_secret(password)


def verify_password(password: str, password_hash: str) -> bool:
    return verify_secret(password, password_hash)


def new_invite_code() -> tuple[str, str, str]:
    code_id = secrets.token_urlsafe(8)
    secret = secrets.token_urlsafe(24)
    return code_id, secret, hash_secret(secret)


def split_invite_code(raw_code: str) -> tuple[str, str] | None:
    code_id, sep, secret = raw_code.rpartition(".")
    if sep != "." or not code_id or not secret:
        return None
    return code_id, secret
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/core/test_security.py -q
```

Expected: all tests in `tests/core/test_security.py` pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add pyproject.toml uv.lock src/genacademy_rag/core/security.py tests/core/test_security.py
git commit -m "feat: add security primitives" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

---

## Task 2: Datastore Migration, Locking, And Hashed Users

**Files:**
- Modify: `src/genacademy_rag/data/datastore.py`
- Modify: `src/genacademy_rag/web/auth.py`
- Expand: `tests/data/test_datastore.py`

- [ ] **Step 1: Add failing migration and auth tests**

Append to `tests/data/test_datastore.py`:

```python
import sqlite3

from genacademy_rag.core.security import is_bcrypt_hash, verify_password


def test_migrate_adds_phase1_columns_and_hashes_plaintext_users(tmp_path):
    db_path = tmp_path / "phase0.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','member')),
            password TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE documents (
            id TEXT PRIMARY KEY, title TEXT, source_type TEXT, repo TEXT, file_path TEXT,
            commit_hash TEXT, filename TEXT, uploaded_by TEXT, status TEXT DEFAULT 'indexed',
            n_chunks INTEGER DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE chunks_meta (
            id TEXT PRIMARY KEY, doc_id TEXT, ordinal INTEGER, page_or_section TEXT,
            line_start INTEGER, line_end INTEGER, char_start INTEGER, char_end INTEGER,
            text_preview TEXT);
        INSERT INTO users(email, role, password) VALUES
            ('admin@genacademy.local', 'admin', 'admin');
        """
    )
    conn.commit()
    conn.close()

    ds = SQLiteDatastore(db_path)
    columns = {row["name"] for row in ds.table_info("documents")}
    assert {"deleted_at", "deleted_by", "stored_path"} <= columns
    assert ds.table_exists("invite_codes")
    assert ds.table_exists("usage_log")
    admin = ds.get_user_by_email("admin@genacademy.local")
    assert admin is not None
    assert is_bcrypt_hash(admin["password"])
    assert verify_password("admin", admin["password"])


def test_seed_users_stores_bcrypt_hashes(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    ds.seed_users()
    admin = ds.get_user_by_email("admin@genacademy.local")
    member = ds.get_user_by_email("member@genacademy.local")
    assert admin is not None and verify_password("admin", admin["password"])
    assert member is not None and verify_password("member", member["password"])


def test_create_user_rejects_duplicate_email(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    password_hash = hash_password("secret")
    created = ds.create_user(email="a@example.com", role="member", password_hash=password_hash)
    duplicate = ds.create_user(email="a@example.com", role="member", password_hash=password_hash)
    assert created is not None
    assert duplicate is None
```

Update imports at the top of `tests/data/test_datastore.py`:

```python
from genacademy_rag.core.security import hash_password, is_bcrypt_hash, verify_password
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
uv run pytest tests/data/test_datastore.py::test_migrate_adds_phase1_columns_and_hashes_plaintext_users tests/data/test_datastore.py::test_seed_users_stores_bcrypt_hashes tests/data/test_datastore.py::test_create_user_rejects_duplicate_email -q
```

Expected: FAIL because `table_info`, `table_exists`, `create_user`, and hashed seeding are not implemented.

- [ ] **Step 3: Implement locked migration helpers**

In `src/genacademy_rag/data/datastore.py`, add imports:

```python
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from genacademy_rag.core.security import hash_password, is_bcrypt_hash
from genacademy_rag.core.types import Chunk
```

Replace `SCHEMA` with fresh Phase-1 schema:

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin','member')),
    password TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, title TEXT, source_type TEXT, repo TEXT, file_path TEXT,
    commit_hash TEXT, filename TEXT, uploaded_by TEXT, status TEXT DEFAULT 'indexed',
    n_chunks INTEGER DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    deleted_at TEXT, deleted_by TEXT, stored_path TEXT);
CREATE TABLE IF NOT EXISTS chunks_meta (
    id TEXT PRIMARY KEY, doc_id TEXT, ordinal INTEGER, page_or_section TEXT,
    line_start INTEGER, line_end INTEGER, char_start INTEGER, char_end INTEGER,
    text_preview TEXT);
CREATE TABLE IF NOT EXISTS invite_codes (
    id TEXT PRIMARY KEY,
    secret_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin','member')),
    created_by TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT,
    used_by TEXT,
    used_at TEXT,
    revoked_at TEXT);
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY, ts TEXT DEFAULT CURRENT_TIMESTAMP,
    user_email TEXT, question TEXT,
    refused INTEGER, confidence INTEGER, used_fallback INTEGER,
    n_citations INTEGER, latency_ms INTEGER);
"""
```

Add utility:

```python
def _utcnow_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
```

Replace `SQLiteDatastore.__init__` and add migration helpers:

```python
class SQLiteDatastore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        document_columns = {row["name"] for row in self.table_info("documents")}
        for name in ("deleted_at", "deleted_by", "stored_path"):
            if name not in document_columns:
                self._conn.execute(f"ALTER TABLE documents ADD COLUMN {name} TEXT")
        rows = self._conn.execute("SELECT email, password FROM users").fetchall()
        for row in rows:
            if not is_bcrypt_hash(row["password"]):
                self._conn.execute(
                    "UPDATE users SET password=? WHERE email=?",
                    (hash_password(row["password"]), row["email"]),
                )

    def table_info(self, table: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            return [dict(row) for row in rows]

    def table_exists(self, table: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            return row is not None
```

Replace `seed_users`, `get_user_by_email`, and add `create_user`:

```python
    def seed_users(self) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO users(email, role, password) VALUES (?,?,?)",
                [
                    ("admin@genacademy.local", "admin", hash_password("admin")),
                    ("member@genacademy.local", "member", hash_password("member")),
                ],
            )
            self._migrate()
            self._conn.commit()

    def get_user_by_email(self, email: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            return dict(row) if row else None

    def create_user(self, *, email: str, role: str, password_hash: str) -> dict | None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO users(email, role, password) VALUES (?,?,?)",
                    (email, role, password_hash),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                self._conn.rollback()
                return None
            return self.get_user_by_email(email)
```

- [ ] **Step 4: Update authentication to verify bcrypt**

Replace `src/genacademy_rag/web/auth.py` with:

```python
"""Thin session auth helpers. Credential verification delegates to pure core security."""
from __future__ import annotations

from genacademy_rag.core.security import verify_password
from genacademy_rag.data.datastore import SQLiteDatastore


def authenticate(datastore: SQLiteDatastore, email: str, password: str) -> dict | None:
    user = datastore.get_user_by_email(email)
    if user and verify_password(password, user["password"]):
        return user
    return None
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/data/test_datastore.py tests/web/test_app.py::test_login_then_ask_renders_cited_answer -q
```

Expected: datastore tests pass and the existing login test still passes with bcrypt-hashed seeded users.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/genacademy_rag/data/datastore.py src/genacademy_rag/web/auth.py tests/data/test_datastore.py
git commit -m "feat: migrate datastore auth schema" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

---

## Task 3: Invite Lifecycle And Atomic Redemption

**Files:**
- Modify: `src/genacademy_rag/data/datastore.py`
- Expand: `tests/data/test_datastore.py`

- [ ] **Step 1: Add failing invite tests**

Append to `tests/data/test_datastore.py`:

```python
from concurrent.futures import ThreadPoolExecutor


def test_generate_redeem_and_reject_reuse_of_invite(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    invite = ds.generate_invite(role="member", created_by="admin@genacademy.local", expires_at=None)
    assert invite["code"].startswith(invite["id"] + ".")
    user = ds.redeem_invite(
        raw_code=invite["code"],
        email="new@example.com",
        password_hash=hash_password("pw"),
    )
    reused = ds.redeem_invite(
        raw_code=invite["code"],
        email="other@example.com",
        password_hash=hash_password("pw"),
    )
    assert user is not None and user["role"] == "member"
    assert reused is None
    listed = ds.list_invites()
    assert listed[0]["status"] == "used"
    assert listed[0]["used_by"] == "new@example.com"


def test_redeem_rejects_bad_secret_revoked_and_expired(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    active = ds.generate_invite(role="member", created_by="admin@genacademy.local", expires_at=None)
    revoked = ds.generate_invite(role="member", created_by="admin@genacademy.local", expires_at=None)
    expired = ds.generate_invite(
        role="member",
        created_by="admin@genacademy.local",
        expires_at="2000-01-01 00:00:00",
    )
    ds.revoke_invite(revoked["id"])
    assert ds.redeem_invite(
        raw_code=active["id"] + ".wrong",
        email="bad@example.com",
        password_hash=hash_password("pw"),
    ) is None
    assert ds.redeem_invite(
        raw_code=revoked["code"],
        email="revoked@example.com",
        password_hash=hash_password("pw"),
    ) is None
    assert ds.redeem_invite(
        raw_code=expired["code"],
        email="expired@example.com",
        password_hash=hash_password("pw"),
    ) is None


def test_concurrent_redeem_allows_exactly_one_winner(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    invite = ds.generate_invite(role="member", created_by="admin@genacademy.local", expires_at=None)

    def redeem(i: int) -> bool:
        user = ds.redeem_invite(
            raw_code=invite["code"],
            email=f"user{i}@example.com",
            password_hash=hash_password("pw"),
        )
        return user is not None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(redeem, range(8)))
    assert results.count(True) == 1
    assert results.count(False) == 7
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/data/test_datastore.py::test_generate_redeem_and_reject_reuse_of_invite tests/data/test_datastore.py::test_redeem_rejects_bad_secret_revoked_and_expired tests/data/test_datastore.py::test_concurrent_redeem_allows_exactly_one_winner -q
```

Expected: FAIL because invite methods are missing.

- [ ] **Step 3: Implement invite methods**

Add imports to `src/genacademy_rag/data/datastore.py`:

```python
from genacademy_rag.core.security import (
    hash_password,
    is_bcrypt_hash,
    new_invite_code,
    split_invite_code,
    verify_secret,
)
```

Add methods to `SQLiteDatastore`:

```python
    def generate_invite(
        self,
        *,
        role: str,
        created_by: str,
        expires_at: str | None,
    ) -> dict:
        code_id, secret, secret_hash = new_invite_code()
        with self._lock:
            self._conn.execute(
                "INSERT INTO invite_codes(id, secret_hash, role, created_by, expires_at) "
                "VALUES (?,?,?,?,?)",
                (code_id, secret_hash, role, created_by, expires_at),
            )
            self._conn.commit()
        return {
            "id": code_id,
            "code": f"{code_id}.{secret}",
            "role": role,
            "created_by": created_by,
            "expires_at": expires_at,
        }

    def list_invites(self) -> list[dict]:
        now = _utcnow_text()
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, role, created_by, created_at, expires_at, used_by, used_at, revoked_at "
                "FROM invite_codes ORDER BY created_at DESC, id DESC"
            ).fetchall()
        out: list[dict] = []
        for row in rows:
            item = dict(row)
            if item["revoked_at"]:
                status = "revoked"
            elif item["used_at"]:
                status = "used"
            elif item["expires_at"] and item["expires_at"] <= now:
                status = "expired"
            else:
                status = "active"
            item["status"] = status
            out.append(item)
        return out

    def revoke_invite(self, code_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE invite_codes SET revoked_at=? WHERE id=? AND used_at IS NULL AND revoked_at IS NULL",
                (_utcnow_text(), code_id),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def redeem_invite(self, *, raw_code: str, email: str, password_hash: str) -> dict | None:
        parts = split_invite_code(raw_code)
        if parts is None:
            return None
        code_id, secret = parts
        now = _utcnow_text()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT * FROM invite_codes WHERE id=?",
                    (code_id,),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return None
                invite = dict(row)
                expired = invite["expires_at"] is not None and invite["expires_at"] <= now
                inactive = invite["used_at"] is not None or invite["revoked_at"] is not None or expired
                if inactive or not verify_secret(secret, invite["secret_hash"]):
                    self._conn.rollback()
                    return None
                cur = self._conn.execute(
                    "UPDATE invite_codes SET used_by=?, used_at=? "
                    "WHERE id=? AND used_at IS NULL AND revoked_at IS NULL "
                    "AND (expires_at IS NULL OR expires_at>?)",
                    (email, now, code_id, now),
                )
                if cur.rowcount != 1:
                    self._conn.rollback()
                    return None
                self._conn.execute(
                    "INSERT INTO users(email, role, password) VALUES (?,?,?)",
                    (email, invite["role"], password_hash),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                self._conn.rollback()
                return None
            except Exception:
                self._conn.rollback()
                raise
            return self.get_user_by_email(email)
```

- [ ] **Step 4: Run invite tests**

Run:

```bash
uv run pytest tests/data/test_datastore.py::test_generate_redeem_and_reject_reuse_of_invite tests/data/test_datastore.py::test_redeem_rejects_bad_secret_revoked_and_expired tests/data/test_datastore.py::test_concurrent_redeem_allows_exactly_one_winner -q
```

Expected: all invite tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/genacademy_rag/data/datastore.py tests/data/test_datastore.py
git commit -m "feat: add atomic invite redemption" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

---

## Task 4: Document Tombstones And Usage Logging

**Files:**
- Modify: `src/genacademy_rag/data/datastore.py`
- Expand: `tests/data/test_datastore.py`

- [ ] **Step 1: Add failing document and usage tests**

Append to `tests/data/test_datastore.py`:

```python
def test_document_delete_is_uploaded_only_and_idempotent(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    ds.add_document(
        doc_id="course",
        title="Course",
        source_type="github",
        uploaded_by=None,
        n_chunks=1,
    )
    ds.add_document(
        doc_id="upload",
        title="Upload",
        source_type="pdf",
        filename="notes.pdf",
        uploaded_by="admin@genacademy.local",
        stored_path=str(tmp_path / "notes.pdf"),
        n_chunks=2,
    )
    ds.add_chunks_meta([_chunk(0, "upload"), _chunk(1, "upload")])
    assert not ds.delete_document("course", deleted_by="admin@genacademy.local")
    assert ds.delete_document("upload", deleted_by="admin@genacademy.local")
    assert ds.delete_document("upload", deleted_by="admin@genacademy.local")
    deleted = ds.get_document("upload")
    assert deleted is not None
    assert deleted["status"] == "deleted"
    assert deleted["deleted_by"] == "admin@genacademy.local"
    assert ds.get_chunks_for_doc("upload") == []


def test_usage_log_round_trip(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    ds.log_query(
        user_email="member@genacademy.local",
        question="What is RAG?",
        refused=False,
        confidence=5,
        used_fallback=False,
        n_citations=2,
        latency_ms=123,
    )
    rows = ds.recent_usage(limit=10)
    assert len(rows) == 1
    assert rows[0]["question"] == "What is RAG?"
    assert rows[0]["refused"] == 0
    assert rows[0]["used_fallback"] == 0
    assert rows[0]["latency_ms"] == 123
```

Update `_chunk` helper in the same file so it accepts a doc id:

```python
def _chunk(i, doc_id="d1"):
    cit = Citation(
        doc_id=doc_id,
        title="README.md",
        source_type="github",
        repo="r",
        file_path="README.md",
        commit_hash="abc123",
        line_start=i,
        line_end=i + 1,
        char_start=i,
        char_end=i + 5,
    )
    return Chunk(
        chunk_id=f"{doc_id}::{i}",
        doc_id=doc_id,
        ordinal=i,
        text=f"chunk {i} preview",
        citation=cit,
    )
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/data/test_datastore.py::test_document_delete_is_uploaded_only_and_idempotent tests/data/test_datastore.py::test_usage_log_round_trip -q
```

Expected: FAIL because `stored_path`, `delete_document`, `log_query`, and `recent_usage` are not fully implemented.

- [ ] **Step 3: Expand document and usage methods**

Update `add_document` signature in `SQLiteDatastore`:

```python
    def add_document(
        self,
        *,
        doc_id,
        title,
        source_type,
        repo=None,
        file_path=None,
        commit_hash=None,
        filename=None,
        uploaded_by=None,
        stored_path=None,
        n_chunks=0,
        status="indexed",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO documents(id,title,source_type,repo,file_path,commit_hash,"
                "filename,uploaded_by,stored_path,n_chunks,status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    title,
                    source_type,
                    repo,
                    file_path,
                    commit_hash,
                    filename,
                    uploaded_by,
                    stored_path,
                    n_chunks,
                    status,
                ),
            )
            self._conn.commit()
```

Wrap existing `get_document`, `add_chunks_meta`, and `get_chunks_for_doc` bodies in `with self._lock:`.

Add methods:

```python
    def list_documents(self, *, include_deleted: bool = True) -> list[dict]:
        where = "" if include_deleted else "WHERE status != 'deleted'"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM documents {where} ORDER BY created_at DESC, id DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_document(self, doc_id: str, *, deleted_by: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if row is None:
                return False
            doc = dict(row)
            if doc["uploaded_by"] is None:
                return False
            self._conn.execute("DELETE FROM chunks_meta WHERE doc_id=?", (doc_id,))
            if doc["status"] != "deleted":
                self._conn.execute(
                    "UPDATE documents SET status='deleted', deleted_at=?, deleted_by=? WHERE id=?",
                    (_utcnow_text(), deleted_by, doc_id),
                )
            self._conn.commit()
            return True

    def log_query(
        self,
        *,
        user_email: str | None,
        question: str,
        refused: bool,
        confidence: int,
        used_fallback: bool,
        n_citations: int,
        latency_ms: int,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage_log(user_email, question, refused, confidence, used_fallback, "
                "n_citations, latency_ms) VALUES (?,?,?,?,?,?,?)",
                (
                    user_email,
                    question,
                    int(refused),
                    confidence,
                    int(used_fallback),
                    n_citations,
                    latency_ms,
                ),
            )
            self._conn.commit()

    def recent_usage(self, *, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM usage_log ORDER BY ts DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
```

- [ ] **Step 4: Add concurrent write safety test**

Append:

```python
def test_concurrent_usage_writes_are_serialized(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")

    def write(i: int) -> None:
        ds.log_query(
            user_email="member@genacademy.local",
            question=f"q{i}",
            refused=i % 2 == 0,
            confidence=3,
            used_fallback=False,
            n_citations=1,
            latency_ms=i,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(40)))
    assert len(ds.recent_usage(limit=100)) == 40
```

- [ ] **Step 5: Run datastore tests**

Run:

```bash
uv run pytest tests/data/test_datastore.py -q
```

Expected: all datastore tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/genacademy_rag/data/datastore.py tests/data/test_datastore.py
git commit -m "feat: add document and usage datastore methods" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

---

## Task 5: VectorStore Delete Seam

**Files:**
- Modify: `src/genacademy_rag/core/vectorstore.py`
- Expand: `tests/core/test_vectorstore.py`

- [ ] **Step 1: Add failing vector delete test**

Append to `tests/core/test_vectorstore.py`:

```python
def test_chroma_delete_doc_removes_only_matching_doc(tmp_path, fake_provider):
    from genacademy_rag.core.types import Chunk, Citation
    from genacademy_rag.core.vectorstore import ChromaStore

    def chunk(doc_id: str, ordinal: int) -> Chunk:
        cit = Citation(doc_id=doc_id, title=doc_id, source_type="pdf")
        return Chunk(
            chunk_id=f"{doc_id}::{ordinal}",
            doc_id=doc_id,
            ordinal=ordinal,
            text=f"{doc_id} text {ordinal}",
            citation=cit,
        )

    store = ChromaStore(persist_dir=tmp_path / "chroma", collection="serving")
    chunks = [chunk("a", 0), chunk("a", 1), chunk("b", 0)]
    store.upsert(chunks, fake_provider.embed([c.text for c in chunks]))
    store.delete_doc("a")
    remaining = store.get_all_chunks()
    assert [c.chunk_id for c in remaining] == ["b::0"]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
uv run pytest tests/core/test_vectorstore.py::test_chroma_delete_doc_removes_only_matching_doc -q
```

Expected: FAIL because `ChromaStore.delete_doc` is missing.

- [ ] **Step 3: Implement protocol and Chroma deletion**

In `src/genacademy_rag/core/vectorstore.py`, add to `VectorStore`:

```python
    def delete_doc(self, doc_id: str) -> None: ...
```

Add to `ChromaStore`:

```python
    def delete_doc(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})
```

- [ ] **Step 4: Run vectorstore tests**

Run:

```bash
uv run pytest tests/core/test_vectorstore.py -q
```

Expected: all vectorstore tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/genacademy_rag/core/vectorstore.py tests/core/test_vectorstore.py
git commit -m "feat: add vector document deletion" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

---

## Task 6: Upload Provenance Through Types, Loader, And Pipeline

**Files:**
- Modify: `src/genacademy_rag/core/types.py`
- Modify: `src/genacademy_rag/core/loaders/pdf_loader.py`
- Modify: `src/genacademy_rag/core/pipeline.py`
- Expand: `tests/core/test_pdf_loader.py`
- Expand: `tests/core/test_ingest_pipeline.py`

- [ ] **Step 1: Add failing provenance tests**

Append to `tests/core/test_pdf_loader.py`:

```python
def test_pdf_loader_preserves_uploaded_by_and_stored_path():
    import io

    from pypdf import PdfWriter

    from genacademy_rag.core.loaders.pdf_loader import load_pdf_bytes

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    raw = io.BytesIO()
    writer.write(raw)
    doc = load_pdf_bytes(
        filename="notes.pdf",
        raw_bytes=raw.getvalue(),
        uploaded_by="admin@genacademy.local",
        stored_path="/tmp/pdf_abcd.pdf",
    )
    assert doc.uploaded_by == "admin@genacademy.local"
    assert doc.stored_path == "/tmp/pdf_abcd.pdf"
```

Append to `tests/core/test_ingest_pipeline.py`:

```python
def test_ingest_pipeline_records_upload_provenance(fake_provider):
    from genacademy_rag.core.chunker import FixedSizeChunker
    from genacademy_rag.core.pipeline import IngestPipeline
    from genacademy_rag.core.types import Document

    calls = {}

    class Store:
        def upsert(self, chunks, embeddings):
            calls["upsert_chunks"] = chunks

    class Datastore:
        def add_document(self, **kwargs):
            calls["document"] = kwargs

        def add_chunks_meta(self, chunks):
            calls["chunks_meta"] = chunks

    doc = Document(
        doc_id="pdf/abc",
        title="notes.pdf",
        source_type="pdf",
        text="Gen Academy notes about retrieval.",
        filename="notes.pdf",
        uploaded_by="admin@genacademy.local",
        stored_path="/tmp/pdf_abc.pdf",
    )
    pipe = IngestPipeline(
        chunker=FixedSizeChunker(chunk_size=50, overlap=5),
        provider=fake_provider,
        store=Store(),
        datastore=Datastore(),
    )
    assert pipe.ingest([doc]) == 1
    assert calls["document"]["uploaded_by"] == "admin@genacademy.local"
    assert calls["document"]["stored_path"] == "/tmp/pdf_abc.pdf"
    assert calls["upsert_chunks"][0].doc_id == "pdf/abc"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/core/test_pdf_loader.py::test_pdf_loader_preserves_uploaded_by_and_stored_path tests/core/test_ingest_pipeline.py::test_ingest_pipeline_records_upload_provenance -q
```

Expected: FAIL because `Document` lacks `uploaded_by` and `stored_path`.

- [ ] **Step 3: Add fields and thread them**

In `src/genacademy_rag/core/types.py`, add to `Document`:

```python
    uploaded_by: str | None = None
    stored_path: str | None = None
```

Change `load_pdf_bytes` signature:

```python
def load_pdf_bytes(
    *,
    filename: str,
    raw_bytes: bytes,
    uploaded_by: str | None = None,
    stored_path: str | None = None,
) -> Document:
```

Change its return:

```python
    return Document(
        doc_id=doc_id,
        title=filename,
        source_type="pdf",
        text=text,
        filename=filename,
        uploaded_by=uploaded_by,
        stored_path=stored_path,
    )
```

In `IngestPipeline.ingest`, pass provenance and write metadata before vector upsert:

```python
            self._datastore.add_document(
                doc_id=doc.doc_id,
                title=doc.title,
                source_type=doc.source_type,
                repo=doc.repo,
                file_path=doc.file_path,
                commit_hash=doc.commit_hash,
                filename=doc.filename,
                uploaded_by=doc.uploaded_by,
                stored_path=doc.stored_path,
                n_chunks=len(chunks),
            )
            self._datastore.add_chunks_meta(chunks)
            self._store.upsert(chunks, embeddings)
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/core/test_pdf_loader.py tests/core/test_ingest_pipeline.py -q
```

Expected: all PDF loader and ingest pipeline tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/genacademy_rag/core/types.py src/genacademy_rag/core/loaders/pdf_loader.py src/genacademy_rag/core/pipeline.py tests/core/test_pdf_loader.py tests/core/test_ingest_pipeline.py
git commit -m "feat: thread upload provenance through ingest" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

---

## Task 7: HybridRetriever Snapshot And Corpus Lock

**Files:**
- Modify: `src/genacademy_rag/core/retriever.py`
- Expand: `tests/core/test_retriever.py`

- [ ] **Step 1: Add failing retriever concurrency tests**

Append to `tests/core/test_retriever.py`:

```python
import threading
import time


def test_reindex_uses_single_snapshot_not_torn_fields(fake_provider):
    old = _chunk(0, "old retrieval text")
    new = _chunk(1, "new Pinecone text")

    class Store:
        def __init__(self):
            self.chunks = [old]

        def query(self, qvec, top_k):
            return [(c.chunk_id, 0.8) for c in self.chunks]

    store = Store()
    retr = HybridRetriever(store=store, provider=fake_provider, all_chunks=[old], top_k=5)
    store.chunks = [new]
    retr.reindex([new])
    results = retr.retrieve("Pinecone")
    assert [r.chunk.chunk_id for r in results] == ["d1::1"]


def test_mutation_lock_prevents_deleted_sparse_orphan(fake_provider):
    keep = _chunk(0, "keep chunk about embeddings")
    delete = _chunk(1, "delete chunk about QLoRA")
    query_entered = threading.Event()
    release_query = threading.Event()

    class Store:
        def __init__(self):
            self.chunks = [keep, delete]

        def query(self, qvec, top_k):
            query_entered.set()
            release_query.wait(timeout=2)
            return [(c.chunk_id, 0.9) for c in self.chunks]

        def delete_doc(self, doc_id):
            self.chunks = [keep]

        def get_all_chunks(self):
            return list(self.chunks)

    store = Store()
    retr = HybridRetriever(store=store, provider=fake_provider, all_chunks=store.get_all_chunks(), top_k=5)
    first_results = []

    def retrieve_before_delete():
        first_results.extend(retr.retrieve("QLoRA"))

    t = threading.Thread(target=retrieve_before_delete)
    t.start()
    assert query_entered.wait(timeout=2)
    mutation_done = threading.Event()

    def mutate():
        retr.mutate_corpus(lambda: (store.delete_doc("d1"), store.get_all_chunks())[1])
        mutation_done.set()

    m = threading.Thread(target=mutate)
    m.start()
    time.sleep(0.05)
    assert not mutation_done.is_set()
    release_query.set()
    t.join(timeout=2)
    m.join(timeout=2)
    assert mutation_done.is_set()
    after = retr.retrieve("QLoRA")
    assert all(r.chunk.chunk_id != "d1::1" for r in after)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/core/test_retriever.py::test_mutation_lock_prevents_deleted_sparse_orphan -q
```

Expected: FAIL because `HybridRetriever.mutate_corpus` is missing.

- [ ] **Step 3: Implement `_Index` and lock**

In `src/genacademy_rag/core/retriever.py`, add imports:

```python
import threading
from collections.abc import Callable
from dataclasses import dataclass
```

Add:

```python
@dataclass(frozen=True)
class _Index:
    ids: tuple[str, ...]
    chunks_by_id: dict[str, Chunk]
    bm25: BM25Okapi


def _build_index(chunks: list[Chunk]) -> _Index:
    chunk_list = list(chunks)
    return _Index(
        ids=tuple(c.chunk_id for c in chunk_list),
        chunks_by_id={c.chunk_id: c for c in chunk_list},
        bm25=BM25Okapi([_tokenize(c.text) for c in chunk_list]),
    )
```

Replace `HybridRetriever` internals:

```python
class HybridRetriever:
    def __init__(
        self,
        *,
        store,
        provider,
        all_chunks: list[Chunk],
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ):
        self._store = store
        self._provider = provider
        self._top_k = top_k
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._corpus_lock = threading.Lock()
        self._index = _build_index(all_chunks)

    def _swap_index_unlocked(self, all_chunks: list[Chunk]) -> None:
        self._index = _build_index(all_chunks)

    def reindex(self, all_chunks: list[Chunk]) -> None:
        with self._corpus_lock:
            self._swap_index_unlocked(all_chunks)

    def mutate_corpus(self, mutation: Callable[[], list[Chunk]]) -> None:
        with self._corpus_lock:
            self._swap_index_unlocked(mutation())

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        with self._corpus_lock:
            index = self._index
            qvec = self._provider.embed([query])[0]
            dense_hits = self._store.query(qvec, top_k=self._candidate_k)
            dense_ids = [cid for cid, _ in dense_hits if cid in index.chunks_by_id]
            sim_by_id = {cid: sim for cid, sim in dense_hits if cid in index.chunks_by_id}
            scores = index.bm25.get_scores(_tokenize(query))
            bm25_order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
            sparse_ids = [index.ids[i] for i in bm25_order][: self._candidate_k]
            fused = rrf_fuse([dense_ids, sparse_ids], k=self._rrf_k)
            ranked = sorted(fused, key=fused.get, reverse=True)[: self._top_k]
            return [
                RetrievedChunk(chunk=index.chunks_by_id[cid], score=sim_by_id.get(cid, 0.0))
                for cid in ranked
            ]
```

- [ ] **Step 4: Run retriever tests**

Run:

```bash
uv run pytest tests/core/test_retriever.py -q
```

Expected: all retriever tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/genacademy_rag/core/retriever.py tests/core/test_retriever.py
git commit -m "feat: serialize hybrid retriever mutations" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

---

## Task 8: Pure Usage Analytics

**Files:**
- Create: `src/genacademy_rag/core/analytics.py`
- Create: `tests/core/test_analytics.py`

- [ ] **Step 1: Add failing analytics tests**

Create `tests/core/test_analytics.py`:

```python
from genacademy_rag.core.analytics import usage_summary


def test_usage_summary_empty_rows():
    summary = usage_summary([])
    assert summary["total_queries"] == 0
    assert summary["refusal_rate"] == 0.0
    assert summary["fallback_rate"] == 0.0
    assert summary["latency_p50_ms"] == 0
    assert summary["latency_p95_ms"] == 0
    assert summary["top_questions"] == []
    assert summary["queries_by_day"] == []


def test_usage_summary_rates_percentiles_top_questions_and_days():
    rows = [
        {"ts": "2026-06-09 10:00:00", "question": "What is RAG?", "refused": 0, "used_fallback": 0, "latency_ms": 100},
        {"ts": "2026-06-09 10:01:00", "question": "What is RAG?", "refused": 1, "used_fallback": 1, "latency_ms": 200},
        {"ts": "2026-06-10 10:00:00", "question": "What is BM25?", "refused": 0, "used_fallback": 0, "latency_ms": 300},
        {"ts": "2026-06-10 10:01:00", "question": "What is CSRF?", "refused": 0, "used_fallback": 1, "latency_ms": 400},
    ]
    summary = usage_summary(rows, top_n=2)
    assert summary["total_queries"] == 4
    assert summary["refusal_rate"] == 0.25
    assert summary["fallback_rate"] == 0.5
    assert summary["latency_p50_ms"] == 250
    assert summary["latency_p95_ms"] == 385
    assert summary["top_questions"] == [
        {"question": "What is RAG?", "count": 2},
        {"question": "What is BM25?", "count": 1},
    ]
    assert summary["queries_by_day"] == [
        {"day": "2026-06-09", "count": 2},
        {"day": "2026-06-10", "count": 2},
    ]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
uv run pytest tests/core/test_analytics.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement analytics**

Create `src/genacademy_rag/core/analytics.py`:

```python
"""Pure usage analytics for the admin dashboard."""
from __future__ import annotations

from collections import Counter
from math import ceil, floor


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lower = floor(pos)
    upper = ceil(pos)
    if lower == upper:
        return int(ordered[lower])
    weight = pos - lower
    return int(round(ordered[lower] * (1 - weight) + ordered[upper] * weight))


def usage_summary(rows: list[dict], *, top_n: int = 5) -> dict:
    total = len(rows)
    if total == 0:
        return {
            "total_queries": 0,
            "refusal_rate": 0.0,
            "fallback_rate": 0.0,
            "latency_p50_ms": 0,
            "latency_p95_ms": 0,
            "top_questions": [],
            "queries_by_day": [],
        }
    refused = sum(1 for row in rows if int(row.get("refused") or 0))
    fallback = sum(1 for row in rows if int(row.get("used_fallback") or 0))
    latencies = [int(row.get("latency_ms") or 0) for row in rows]
    questions = Counter(str(row.get("question") or "") for row in rows)
    days = Counter(str(row.get("ts") or "")[:10] for row in rows)
    return {
        "total_queries": total,
        "refusal_rate": refused / total,
        "fallback_rate": fallback / total,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "top_questions": [
            {"question": question, "count": count}
            for question, count in questions.most_common(top_n)
        ],
        "queries_by_day": [
            {"day": day, "count": days[day]}
            for day in sorted(days)
            if day
        ],
    }
```

- [ ] **Step 4: Run analytics tests**

Run:

```bash
uv run pytest tests/core/test_analytics.py -q
```

Expected: all analytics tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/genacademy_rag/core/analytics.py tests/core/test_analytics.py
git commit -m "feat: add usage analytics" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

---

## Task 9: Web CSRF, Signup, And Invite Admin

**Files:**
- Modify: `src/genacademy_rag/web/app.py`
- Modify: `src/genacademy_rag/web/templates/login.html`
- Modify: `src/genacademy_rag/web/templates/chat.html`
- Create: `src/genacademy_rag/web/templates/signup.html`
- Create: `src/genacademy_rag/web/templates/admin_invites.html`
- Expand: `tests/web/test_app.py`

- [ ] **Step 1: Add failing web auth tests**

Add helper to `tests/web/test_app.py`:

```python
import re


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def _login(client, email="member@genacademy.local", password="member"):
    page = client.get("/login")
    return client.post(
        "/login",
        data={"email": email, "password": password, "csrf_token": _csrf(page.text)},
    )
```

Update every existing login POST in `tests/web/test_app.py` to use `_login(...)`. Update every existing `/ask` POST to fetch `page = c.get("/")` and include `csrf_token: _csrf(page.text)`.

Append tests:

```python
def test_signup_redeems_invite_and_logs_in(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c, "admin@genacademy.local", "admin")
    r = c.get("/admin/invites")
    token = _csrf(r.text)
    generated = c.post(
        "/admin/invites",
        data={"role": "member", "expires_days": "7", "csrf_token": token},
    )
    code = re.search(r"Invite code: ([^<]+)<", generated.text).group(1)
    signup = c.get("/signup")
    signup_token = _csrf(signup.text)
    created = c.post(
        "/signup",
        data={
            "email": "new@example.com",
            "password": "secret",
            "code": code,
            "csrf_token": signup_token,
        },
        follow_redirects=False,
    )
    assert created.status_code == 303
    home = c.get("/")
    assert "Ask the cohort materials" in home.text


def test_admin_routes_block_member(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    assert c.get("/admin/invites", follow_redirects=False).status_code == 403


def test_csrf_required_for_invite_generation(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c, "admin@genacademy.local", "admin")
    r = c.post("/admin/invites", data={"role": "member", "expires_days": "7"})
    assert r.status_code == 403
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/web/test_app.py::test_signup_redeems_invite_and_logs_in tests/web/test_app.py::test_admin_routes_block_member tests/web/test_app.py::test_csrf_required_for_invite_generation -q
```

Expected: FAIL because routes/templates are missing.

- [ ] **Step 3: Add CSRF and admin helpers in `web/app.py`**

Inside `create_app`, add imports at top:

```python
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from dataclasses import replace
```

Inside `create_app`, add helpers:

```python
    def csrf_token(request: Request) -> str:
        token = request.session.get("csrf_token")
        if not token:
            token = secrets.token_urlsafe(32)
            request.session["csrf_token"] = token
        return token

    def csrf_context(request: Request, extra: dict | None = None) -> dict:
        context = {"csrf_token": csrf_token(request)}
        if extra:
            context.update(extra)
        return context

    def valid_csrf(request: Request, token: str | None) -> bool:
        expected = request.session.get("csrf_token")
        return bool(expected and token and hmac.compare_digest(expected, token))

    def csrf_forbidden() -> HTMLResponse:
        return HTMLResponse("Forbidden", status_code=403)

    def require_admin(request: Request) -> dict | None:
        email = request.session.get("email")
        if not email:
            return None
        user = datastore.get_user_by_email(email)
        if not user or user["role"] != "admin":
            return None
        request.session["role"] = user["role"]
        return user
```

Change login GET and POST to include/validate CSRF:

```python
    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request):
        return TEMPLATES.TemplateResponse(request, "login.html", csrf_context(request, {"error": None}))

    @app.post("/login")
    def login(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        csrf_token_value: str = Form(..., alias="csrf_token"),
    ):
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        user = authenticate(datastore, email, password)
        if not user:
            return TEMPLATES.TemplateResponse(
                request,
                "login.html",
                csrf_context(request, {"error": "Invalid credentials"}),
                status_code=401,
            )
        request.session["email"] = user["email"]
        request.session["role"] = user["role"]
        return RedirectResponse("/", status_code=303)
```

- [ ] **Step 4: Add signup and invite routes**

Inside `create_app`, add:

```python
    @app.get("/signup", response_class=HTMLResponse)
    def signup_form(request: Request):
        return TEMPLATES.TemplateResponse(request, "signup.html", csrf_context(request, {"error": None}))

    @app.post("/signup")
    def signup(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        code: str = Form(...),
        csrf_token_value: str = Form(..., alias="csrf_token"),
    ):
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        from genacademy_rag.core.security import hash_password

        user = datastore.redeem_invite(
            raw_code=code,
            email=email,
            password_hash=hash_password(password),
        )
        if not user:
            return TEMPLATES.TemplateResponse(
                request,
                "signup.html",
                csrf_context(request, {"error": "Invalid or expired code"}),
                status_code=400,
            )
        request.session["email"] = user["email"]
        request.session["role"] = user["role"]
        return RedirectResponse("/", status_code=303)

    @app.get("/admin/invites", response_class=HTMLResponse)
    def admin_invites(request: Request):
        admin = require_admin(request)
        if not admin:
            return HTMLResponse("Forbidden", status_code=403)
        return TEMPLATES.TemplateResponse(
            request,
            "admin_invites.html",
            csrf_context(request, {"invites": datastore.list_invites(), "new_code": None}),
        )

    @app.post("/admin/invites", response_class=HTMLResponse)
    def generate_invite(
        request: Request,
        role: str = Form(...),
        expires_days: int = Form(7),
        csrf_token_value: str = Form(..., alias="csrf_token"),
    ):
        admin = require_admin(request)
        if not admin:
            return HTMLResponse("Forbidden", status_code=403)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        expires_at = None
        if expires_days > 0:
            expires_at = (
                datetime.now(timezone.utc) + timedelta(days=expires_days)
            ).strftime("%Y-%m-%d %H:%M:%S")
        invite = datastore.generate_invite(
            role=role,
            created_by=admin["email"],
            expires_at=expires_at,
        )
        return TEMPLATES.TemplateResponse(
            request,
            "admin_invites.html",
            csrf_context(request, {"invites": datastore.list_invites(), "new_code": invite["code"]}),
        )

    @app.post("/admin/invites/{code_id}/revoke")
    def revoke_invite(
        request: Request,
        code_id: str,
        csrf_token_value: str = Form(..., alias="csrf_token"),
    ):
        if not require_admin(request):
            return HTMLResponse("Forbidden", status_code=403)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        datastore.revoke_invite(code_id)
        return RedirectResponse("/admin/invites", status_code=303)
```

- [ ] **Step 5: Update templates**

In `login.html`, add hidden CSRF input inside the form:

```html
  <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
```

In `chat.html`, add hidden CSRF input inside the ask form:

```html
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
```

Create `signup.html`:

```html
<!doctype html><html><head><meta charset="utf-8"><title>GenAcademy RAG — Sign up</title>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-50 min-h-screen flex items-center justify-center">
<form method="post" action="/signup" class="bg-white p-8 rounded-xl shadow w-96 space-y-4">
  <h1 class="text-xl font-semibold">Create account</h1>
  {% if error %}<p class="text-red-600 text-sm">{{ error }}</p>{% endif %}
  <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
  <input name="email" placeholder="email" class="w-full border rounded px-3 py-2">
  <input name="password" type="password" placeholder="password" class="w-full border rounded px-3 py-2">
  <input name="code" placeholder="invite code" class="w-full border rounded px-3 py-2">
  <button class="w-full bg-slate-900 text-white rounded py-2">Sign up</button>
</form></body></html>
```

Create `admin_invites.html`:

```html
<!doctype html><html><head><meta charset="utf-8"><title>Admin invites</title>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-50 min-h-screen">
<main class="max-w-4xl mx-auto p-6 space-y-6">
  <h1 class="text-2xl font-semibold">Invites</h1>
  {% if new_code %}<p class="bg-emerald-50 border border-emerald-200 rounded p-3">Invite code: {{ new_code }}</p>{% endif %}
  <form method="post" action="/admin/invites" class="flex gap-2 items-end">
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
    <label class="text-sm">Role
      <select name="role" class="block border rounded px-3 py-2">
        <option value="member">member</option>
        <option value="admin">admin</option>
      </select>
    </label>
    <label class="text-sm">Expires in days
      <input name="expires_days" value="7" class="block border rounded px-3 py-2 w-32">
    </label>
    <button class="bg-slate-900 text-white rounded px-4 py-2">Generate</button>
  </form>
  <table class="w-full bg-white rounded shadow text-sm">
    <thead><tr><th class="text-left p-2">ID</th><th class="text-left p-2">Role</th><th class="text-left p-2">Status</th><th class="text-left p-2">Used by</th><th class="p-2"></th></tr></thead>
    <tbody>
    {% for invite in invites %}
      <tr class="border-t">
        <td class="p-2">{{ invite.id }}</td>
        <td class="p-2">{{ invite.role }}</td>
        <td class="p-2">{{ invite.status }}</td>
        <td class="p-2">{{ invite.used_by or "" }}</td>
        <td class="p-2 text-right">
          {% if invite.status == "active" %}
          <form method="post" action="/admin/invites/{{ invite.id }}/revoke">
            <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
            <button class="border rounded px-3 py-1">Revoke</button>
          </form>
          {% endif %}
        </td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</main></body></html>
```

- [ ] **Step 6: Run web auth tests**

Run:

```bash
uv run pytest tests/web/test_app.py::test_signup_redeems_invite_and_logs_in tests/web/test_app.py::test_admin_routes_block_member tests/web/test_app.py::test_csrf_required_for_invite_generation -q
```

Expected: all three tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/genacademy_rag/web/app.py src/genacademy_rag/web/templates/login.html src/genacademy_rag/web/templates/chat.html src/genacademy_rag/web/templates/signup.html src/genacademy_rag/web/templates/admin_invites.html tests/web/test_app.py
git commit -m "feat: add invite signup routes" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

---

## Task 10: Admin Documents Upload, Delete, And Reindex

**Files:**
- Modify: `src/genacademy_rag/web/app.py`
- Create: `src/genacademy_rag/web/templates/admin_documents.html`
- Expand: `tests/web/test_app.py`

- [ ] **Step 1: Add failing admin document tests**

Append to `tests/web/test_app.py`:

```python
def test_upload_uses_content_hash_path_and_avoids_filename_collision(monkeypatch, tmp_path):
    import io
    from pathlib import Path

    from pypdf import PdfWriter
    from starlette.testclient import TestClient

    from genacademy_rag.core.chunker import FixedSizeChunker
    from genacademy_rag.core.pipeline import IngestPipeline
    from genacademy_rag.core.vectorstore import ChromaStore
    from genacademy_rag.data.datastore import SQLiteDatastore
    from genacademy_rag.web.app import create_app
    from tests.conftest import FakeModelProvider

    monkeypatch.setenv("GENACADEMY_SESSION_SECRET", "test-secret")
    provider = FakeModelProvider()
    datastore = SQLiteDatastore(tmp_path / "t.sqlite")
    serving = ChromaStore(persist_dir=tmp_path / "chroma", collection="serving")

    class Retriever:
        def retrieve(self, q):
            return []

        def mutate_corpus(self, mutation):
            mutation()

    pipe = IngestPipeline(
        chunker=FixedSizeChunker(chunk_size=100, overlap=10),
        provider=provider,
        store=serving,
        datastore=datastore,
    )

    def ingest_upload(doc):
        pipe.ingest([doc])

    app = create_app(
        retriever=Retriever(),
        provider=provider,
        datastore=datastore,
        ingest_upload=ingest_upload,
        serving_store=serving,
        uploads_dir=tmp_path / "uploads",
    )
    c = TestClient(app)
    _login(c, "admin@genacademy.local", "admin")
    page = c.get("/admin/documents")
    token = _csrf(page.text)
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    first = io.BytesIO()
    writer.write(first)
    writer = PdfWriter()
    writer.add_blank_page(width=210, height=210)
    second = io.BytesIO()
    writer.write(second)
    c.post("/upload", data={"csrf_token": token}, files={"file": ("same.pdf", first.getvalue(), "application/pdf")})
    c.post("/upload", data={"csrf_token": token}, files={"file": ("same.pdf", second.getvalue(), "application/pdf")})
    stored = list((tmp_path / "uploads").glob("*.pdf"))
    assert len(stored) == 2
    assert all(Path(p).name != "same.pdf" for p in stored)


def test_delete_route_removes_serving_doc_and_leaves_eval_pristine(monkeypatch, tmp_path):
    from genacademy_rag.web.app import create_app
    from genacademy_rag.data.datastore import SQLiteDatastore
    from tests.conftest import FakeModelProvider

    monkeypatch.setenv("GENACADEMY_SESSION_SECRET", "test-secret")
    datastore = SQLiteDatastore(tmp_path / "t.sqlite")
    provider = FakeModelProvider()
    deleted = []
    reindexed = []

    class Retriever:
        def retrieve(self, q):
            return []

        def mutate_corpus(self, mutation):
            reindexed.append(mutation())

    class Serving:
        def delete_doc(self, doc_id):
            deleted.append(doc_id)

        def get_all_chunks(self):
            return []

    datastore.add_document(
        doc_id="pdf/abc",
        title="notes.pdf",
        source_type="pdf",
        filename="notes.pdf",
        uploaded_by="admin@genacademy.local",
        stored_path=str(tmp_path / "notes.pdf"),
        n_chunks=1,
    )
    app = create_app(
        retriever=Retriever(),
        provider=provider,
        datastore=datastore,
        serving_store=Serving(),
        uploads_dir=tmp_path / "uploads",
    )
    c = TestClient(app)
    _login(c, "admin@genacademy.local", "admin")
    page = c.get("/admin/documents")
    token = _csrf(page.text)
    r = c.post(
        "/admin/documents/delete",
        data={"doc_id": "pdf/abc", "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert deleted == ["pdf/abc"]
    assert reindexed == [[]]
    assert datastore.get_document("pdf/abc")["status"] == "deleted"
```

During implementation, remove unused imports from the new test if ruff flags them.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/web/test_app.py::test_upload_uses_content_hash_path_and_avoids_filename_collision tests/web/test_app.py::test_delete_route_removes_serving_doc_and_leaves_eval_pristine -q
```

Expected: FAIL because document routes and content-hash upload storage are not implemented.

- [ ] **Step 3: Extend app injection signature**

Change `create_app` signature:

```python
def create_app(
    *,
    retriever,
    provider,
    datastore,
    ingest_upload=None,
    serving_store=None,
    uploads_dir=None,
) -> FastAPI:
```

Keep existing tests working by making `serving_store` optional.

- [ ] **Step 4: Update home/chat context and `/ask` CSRF validation**

Change home:

```python
        return TEMPLATES.TemplateResponse(
            request,
            "chat.html",
            csrf_context(request, {"result": None, "question": None}),
        )
```

Change ask signature and validation:

```python
    @app.post("/ask", response_class=HTMLResponse)
    def ask(
        request: Request,
        question: str = Form(...),
        csrf_token_value: str = Form(..., alias="csrf_token"),
    ):
        if not current_user(request):
            return RedirectResponse("/login", status_code=303)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        result = qp.answer(question)
        return TEMPLATES.TemplateResponse(
            request,
            "chat.html",
            csrf_context(request, {"result": result, "question": question}),
        )
```

- [ ] **Step 5: Make upload CSRF-protected and content-hash stored**

Replace upload route body:

```python
    @app.post("/upload")
    async def upload(
        request: Request,
        file: UploadFile = File(...),
        csrf_token_value: str = Form(..., alias="csrf_token"),
    ):
        admin = require_admin(request)
        if not admin or ingest_upload is None:
            return RedirectResponse("/login", status_code=303)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        raw = await file.read()
        safe_name = Path(file.filename or "upload.pdf").name
        from genacademy_rag.core.loaders.pdf_loader import load_pdf_bytes

        doc = load_pdf_bytes(filename=safe_name, raw_bytes=raw, uploaded_by=admin["email"])
        suffix = Path(safe_name).suffix.lower() or ".pdf"
        stored_name = doc.doc_id.replace("/", "_") + suffix
        stored_path = None
        if uploads_dir is not None:
            uploads_dir.mkdir(parents=True, exist_ok=True)
            stored_path = uploads_dir / stored_name
            stored_path.write_bytes(raw)
        doc = replace(doc, stored_path=str(stored_path) if stored_path else None)
        try:
            ingest_upload(doc)
        except Exception:
            if stored_path is not None:
                stored_path.unlink(missing_ok=True)
            raise
        return RedirectResponse("/admin/documents", status_code=303)
```

- [ ] **Step 6: Add admin documents routes**

Inside `create_app`, add:

```python
    @app.get("/admin/documents", response_class=HTMLResponse)
    def admin_documents(request: Request):
        if not require_admin(request):
            return HTMLResponse("Forbidden", status_code=403)
        return TEMPLATES.TemplateResponse(
            request,
            "admin_documents.html",
            csrf_context(request, {"documents": datastore.list_documents()}),
        )

    @app.post("/admin/documents/delete")
    def delete_document(
        request: Request,
        doc_id: str = Form(...),
        csrf_token_value: str = Form(..., alias="csrf_token"),
    ):
        admin = require_admin(request)
        if not admin:
            return HTMLResponse("Forbidden", status_code=403)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        doc = datastore.get_document(doc_id)
        if not doc or doc["uploaded_by"] is None:
            return HTMLResponse("Forbidden", status_code=403)
        if serving_store is not None:
            def mutation():
                serving_store.delete_doc(doc_id)
                return serving_store.get_all_chunks()

            retriever.mutate_corpus(mutation)
        if doc.get("stored_path"):
            Path(doc["stored_path"]).unlink(missing_ok=True)
        datastore.delete_document(doc_id, deleted_by=admin["email"])
        return RedirectResponse("/admin/documents", status_code=303)

    @app.post("/admin/documents/reindex")
    def reindex_documents(
        request: Request,
        csrf_token_value: str = Form(..., alias="csrf_token"),
    ):
        if not require_admin(request):
            return HTMLResponse("Forbidden", status_code=403)
        if not valid_csrf(request, csrf_token_value):
            return csrf_forbidden()
        if serving_store is not None:
            retriever.mutate_corpus(lambda: serving_store.get_all_chunks())
        return RedirectResponse("/admin/documents", status_code=303)
```

- [ ] **Step 7: Add documents template**

Create `src/genacademy_rag/web/templates/admin_documents.html`:

```html
<!doctype html><html><head><meta charset="utf-8"><title>Admin documents</title>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-50 min-h-screen">
<main class="max-w-5xl mx-auto p-6 space-y-6">
  <h1 class="text-2xl font-semibold">Documents</h1>
  <form method="post" action="/upload" enctype="multipart/form-data" class="bg-white rounded shadow p-4 flex gap-3 items-end">
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
    <label class="text-sm flex-1">PDF
      <input name="file" type="file" class="block w-full border rounded px-3 py-2">
    </label>
    <button class="bg-slate-900 text-white rounded px-4 py-2">Upload</button>
  </form>
  <form method="post" action="/admin/documents/reindex">
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
    <button class="border rounded px-4 py-2 bg-white">Re-index serving corpus</button>
  </form>
  <table class="w-full bg-white rounded shadow text-sm">
    <thead><tr><th class="text-left p-2">Title</th><th class="text-left p-2">Type</th><th class="text-left p-2">Status</th><th class="text-left p-2">Chunks</th><th class="text-left p-2">Uploaded by</th><th class="p-2"></th></tr></thead>
    <tbody>
    {% for doc in documents %}
      <tr class="border-t">
        <td class="p-2">{{ doc.title }}</td>
        <td class="p-2">{{ doc.source_type }}</td>
        <td class="p-2">{{ doc.status }}</td>
        <td class="p-2">{{ doc.n_chunks }}</td>
        <td class="p-2">{{ doc.uploaded_by or "" }}</td>
        <td class="p-2 text-right">
          {% if doc.uploaded_by and doc.status != "deleted" %}
          <form method="post" action="/admin/documents/delete">
            <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
            <input type="hidden" name="doc_id" value="{{ doc.id }}">
            <button class="border rounded px-3 py-1">Delete</button>
          </form>
          {% endif %}
        </td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</main></body></html>
```

- [ ] **Step 8: Wire default app serving mutations**

In `build_default_app`, change `ingest_upload`:

```python
    def ingest_upload(doc):
        def mutation():
            pipe.ingest([doc])
            return serving.get_all_chunks()

        retriever.mutate_corpus(mutation)
```

Pass `serving_store=serving` to `create_app`:

```python
    return create_app(
        retriever=retriever,
        provider=provider,
        datastore=datastore,
        ingest_upload=ingest_upload,
        serving_store=serving,
        uploads_dir=uploads_dir,
    )
```

- [ ] **Step 9: Run admin document tests**

Run:

```bash
uv run pytest tests/web/test_app.py::test_upload_uses_content_hash_path_and_avoids_filename_collision tests/web/test_app.py::test_delete_route_removes_serving_doc_and_leaves_eval_pristine tests/web/test_app.py::test_admin_upload_is_searchable_and_eval_stays_pristine -q
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit**

Run:

```bash
git add src/genacademy_rag/web/app.py src/genacademy_rag/web/templates/admin_documents.html tests/web/test_app.py
git commit -m "feat: add admin document management" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

---

## Task 11: Ask Usage Logging And Dashboard

**Files:**
- Modify: `src/genacademy_rag/web/app.py`
- Create: `src/genacademy_rag/web/templates/admin_dashboard.html`
- Expand: `tests/web/test_app.py`

- [ ] **Step 1: Add failing ask/dashboard tests**

Append to `tests/web/test_app.py`:

```python
def test_ask_requires_csrf_and_writes_usage_row(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c)
    forbidden = c.post("/ask", data={"question": "what is RAG?"})
    assert forbidden.status_code == 403
    page = c.get("/")
    ok = c.post("/ask", data={"question": "what is RAG?", "csrf_token": _csrf(page.text)})
    assert ok.status_code == 200
    rows = c.app.state.datastore.recent_usage(limit=10)
    assert len(rows) == 1
    assert rows[0]["question"] == "what is RAG?"
    assert rows[0]["user_email"] == "member@genacademy.local"


def test_dashboard_renders_usage_summary(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c, "admin@genacademy.local", "admin")
    c.app.state.datastore.log_query(
        user_email="member@genacademy.local",
        question="What is RAG?",
        refused=False,
        confidence=5,
        used_fallback=False,
        n_citations=2,
        latency_ms=100,
    )
    r = c.get("/admin/dashboard")
    assert r.status_code == 200
    assert "Total queries" in r.text
    assert "What is RAG?" in r.text
    assert "<svg" in r.text
```

During implementation, remove the local `SQLiteDatastore` import from this test if ruff flags it as unused.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/web/test_app.py::test_ask_requires_csrf_and_writes_usage_row tests/web/test_app.py::test_dashboard_renders_usage_summary -q
```

Expected: FAIL because `/ask` does not log usage and `/admin/dashboard` is missing.

- [ ] **Step 3: Expose datastore for tests**

Inside `create_app` after `app = FastAPI()`:

```python
    app.state.datastore = datastore
```

- [ ] **Step 4: Time `/ask` and log usage**

Replace ask body after CSRF validation:

```python
        start = time.perf_counter()
        result = qp.answer(question)
        latency_ms = int((time.perf_counter() - start) * 1000)
        datastore.log_query(
            user_email=current_user(request),
            question=question,
            refused=result.refused,
            confidence=result.confidence,
            used_fallback=result.used_fallback,
            n_citations=len(result.citations),
            latency_ms=latency_ms,
        )
        return TEMPLATES.TemplateResponse(
            request,
            "chat.html",
            csrf_context(request, {"result": result, "question": question}),
        )
```

- [ ] **Step 5: Add dashboard route**

Inside `create_app`, add:

```python
    @app.get("/admin/dashboard", response_class=HTMLResponse)
    def admin_dashboard(request: Request):
        if not require_admin(request):
            return HTMLResponse("Forbidden", status_code=403)
        from genacademy_rag.core.analytics import usage_summary

        rows = datastore.recent_usage(limit=500)
        summary = usage_summary(rows)
        return TEMPLATES.TemplateResponse(
            request,
            "admin_dashboard.html",
            csrf_context(request, {"summary": summary, "rows": rows}),
        )
```

- [ ] **Step 6: Add dashboard template**

Create `src/genacademy_rag/web/templates/admin_dashboard.html`:

```html
<!doctype html><html><head><meta charset="utf-8"><title>Admin dashboard</title>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-50 min-h-screen">
<main class="max-w-5xl mx-auto p-6 space-y-6">
  <h1 class="text-2xl font-semibold">Dashboard</h1>
  <section class="grid grid-cols-2 md:grid-cols-5 gap-3">
    <div class="bg-white rounded shadow p-3"><p class="text-xs text-slate-500">Total queries</p><p class="text-2xl">{{ summary.total_queries }}</p></div>
    <div class="bg-white rounded shadow p-3"><p class="text-xs text-slate-500">Refusal rate</p><p class="text-2xl">{{ "%.0f"|format(summary.refusal_rate * 100) }}%</p></div>
    <div class="bg-white rounded shadow p-3"><p class="text-xs text-slate-500">Fallback rate</p><p class="text-2xl">{{ "%.0f"|format(summary.fallback_rate * 100) }}%</p></div>
    <div class="bg-white rounded shadow p-3"><p class="text-xs text-slate-500">p50</p><p class="text-2xl">{{ summary.latency_p50_ms }} ms</p></div>
    <div class="bg-white rounded shadow p-3"><p class="text-xs text-slate-500">p95</p><p class="text-2xl">{{ summary.latency_p95_ms }} ms</p></div>
  </section>
  <section class="bg-white rounded shadow p-4">
    <h2 class="font-semibold mb-3">Queries by day</h2>
    <svg width="100%" height="120" role="img">
      {% for row in summary.queries_by_day %}
      <rect x="{{ loop.index0 * 70 }}" y="{{ 110 - (row.count * 20) }}" width="48" height="{{ row.count * 20 }}" fill="#0f172a"></rect>
      <text x="{{ loop.index0 * 70 }}" y="118" font-size="10">{{ row.day[5:] }}</text>
      {% endfor %}
    </svg>
  </section>
  <section class="bg-white rounded shadow p-4">
    <h2 class="font-semibold mb-3">Top questions</h2>
    <ul class="space-y-1">
      {% for row in summary.top_questions %}
      <li class="flex justify-between border-b py-1"><span>{{ row.question }}</span><span>{{ row.count }}</span></li>
      {% endfor %}
    </ul>
  </section>
  <section class="bg-white rounded shadow p-4">
    <h2 class="font-semibold mb-3">Recent usage</h2>
    <table class="w-full text-sm">
      <thead><tr><th class="text-left p-2">Time</th><th class="text-left p-2">User</th><th class="text-left p-2">Question</th><th class="text-left p-2">Refused</th><th class="text-left p-2">Latency</th></tr></thead>
      <tbody>
      {% for row in rows %}
        <tr class="border-t"><td class="p-2">{{ row.ts }}</td><td class="p-2">{{ row.user_email }}</td><td class="p-2">{{ row.question }}</td><td class="p-2">{{ row.refused }}</td><td class="p-2">{{ row.latency_ms }} ms</td></tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
</main></body></html>
```

- [ ] **Step 7: Run dashboard tests**

Run:

```bash
uv run pytest tests/web/test_app.py::test_ask_requires_csrf_and_writes_usage_row tests/web/test_app.py::test_dashboard_renders_usage_summary -q
```

Expected: both tests pass.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/genacademy_rag/web/app.py src/genacademy_rag/web/templates/admin_dashboard.html tests/web/test_app.py
git commit -m "feat: add usage logging dashboard" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

---

## Task 12: Full Regression, Eval Reproducibility, And Demo Path

**Files:**
- Modify only files required by failures discovered in this task.

- [ ] **Step 1: Run unit and route suite**

Run:

```bash
uv run pytest
```

Expected: all non-integration tests pass; the integration test remains deselected.

- [ ] **Step 2: Run ruff**

Run:

```bash
uv run ruff check src tests scripts
```

Expected: `All checks passed!`

- [ ] **Step 3: Run deterministic retrieval eval**

Run:

```bash
uv run python scripts/eval_retrieval.py
```

Expected: the `RETRIEVAL EVAL` summary line matches the Task 0 baseline exactly. If it differs, inspect `src/genacademy_rag/core/retriever.py` first; Phase 1 must not change eval retrieval behavior.

- [ ] **Step 4: Run a TestClient demo flow**

Add a final test to `tests/web/test_app.py` only if the existing focused tests do not already cover the full flow in one path:

```python
def test_phase1_demo_flow(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    _login(c, "admin@genacademy.local", "admin")
    invites = c.get("/admin/invites")
    generated = c.post(
        "/admin/invites",
        data={"role": "member", "expires_days": "7", "csrf_token": _csrf(invites.text)},
    )
    code = re.search(r"Invite code: ([^<]+)<", generated.text).group(1)
    signup = c.get("/signup")
    created = c.post(
        "/signup",
        data={"email": "cohort@example.com", "password": "secret", "code": code, "csrf_token": _csrf(signup.text)},
        follow_redirects=False,
    )
    assert created.status_code == 303
    chat = c.get("/")
    asked = c.post("/ask", data={"question": "what is RAG?", "csrf_token": _csrf(chat.text)})
    assert asked.status_code == 200
    _login(c, "admin@genacademy.local", "admin")
    dashboard = c.get("/admin/dashboard")
    assert "what is RAG?" in dashboard.text
```

Run:

```bash
uv run pytest tests/web/test_app.py::test_phase1_demo_flow -q
```

Expected: demo-flow test passes.

- [ ] **Step 5: Commit final regression fixes or demo-flow test**

If Step 4 added the test or Step 1-3 required fixes, run:

```bash
git add src tests scripts
git commit -m "test: cover phase1 demo flow" -m "Co-Authored-By: Codex <noreply@openai.com>"
```

If no files changed, do not create an empty commit.

---

## Self-Review Checklist

- Scope coverage:
  - RBAC invite signup: Tasks 1-3 and 9.
  - Admin content upload/list/delete/reindex: Tasks 5-7 and 10.
  - Usage logging/dashboard: Tasks 4, 8, and 11.
  - Eval immutability: Tasks 10 and 12.
  - Refusal path: `/ask` continues through `QueryPipeline.answer()` in Tasks 10-11.
  - Pure core/thin view: CSRF/session/template code appears only in `web/`; `security.py` and `analytics.py` stay pure.
  - Concurrency: datastore `RLock` in Tasks 2-4; corpus lock and mutation helper in Task 7; route lock ordering in Tasks 10-11.
- Type consistency:
  - Invite methods use `generate_invite`, `list_invites`, `revoke_invite`, `redeem_invite`.
  - Retriever mutation API is `mutate_corpus(lambda: list_of_chunks)`.
  - Stored upload path lives on `Document.stored_path` and `documents.stored_path`.
- Verification:
  - Focused tests after every task.
  - Full `uv run pytest`, `uv run ruff check src tests scripts`, and `uv run python scripts/eval_retrieval.py` at the end.

## Approval Checkpoint

Pause here. Do not implement Phase 1 code until the user approves this plan.
