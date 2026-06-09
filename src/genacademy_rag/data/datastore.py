"""Datastore seam (SQLite, Phase 0). Holds users, documents, chunks_meta. Vectors live in Chroma;
everything relational here. Phase-2 swap target: Postgres. usage_log is Phase 1."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from genacademy_rag.core.security import hash_password, is_bcrypt_hash
from genacademy_rag.core.types import Chunk

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


def _utcnow_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Datastore(Protocol):
    def seed_users(self) -> None: ...
    def get_user_by_email(self, email: str) -> dict | None: ...
    def create_user(self, *, email: str, role: str, password_hash: str) -> dict | None: ...
    def add_document(self, **kwargs) -> None: ...
    def add_chunks_meta(self, chunks: list[Chunk]) -> None: ...


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
        n_chunks=0,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO documents(id,title,source_type,repo,file_path,commit_hash,"
                "filename,uploaded_by,n_chunks) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    title,
                    source_type,
                    repo,
                    file_path,
                    commit_hash,
                    filename,
                    uploaded_by,
                    n_chunks,
                ),
            )
            self._conn.commit()

    def get_document(self, doc_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def add_chunks_meta(self, chunks: list[Chunk]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks_meta(id,doc_id,ordinal,page_or_section,line_start,"
                "line_end,char_start,char_end,text_preview) VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.ordinal,
                        c.citation.page_or_section,
                        c.citation.line_start,
                        c.citation.line_end,
                        c.citation.char_start,
                        c.citation.char_end,
                        c.text[:200],
                    )
                    for c in chunks
                ],
            )
            self._conn.commit()

    def get_chunks_for_doc(self, doc_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks_meta WHERE doc_id=? ORDER BY ordinal", (doc_id,)
            ).fetchall()
            return [dict(r) for r in rows]
