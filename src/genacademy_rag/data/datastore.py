"""Datastore seam (SQLite, Phase 0). Holds users, documents, chunks_meta. Vectors live in Chroma;
everything relational here. Phase-2 swap target: Postgres. usage_log is Phase 1."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

from genacademy_rag.core.types import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin','member')),
    password TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, title TEXT, source_type TEXT, repo TEXT, file_path TEXT,
    commit_hash TEXT, filename TEXT, uploaded_by TEXT, status TEXT DEFAULT 'indexed',
    n_chunks INTEGER DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS chunks_meta (
    id TEXT PRIMARY KEY, doc_id TEXT, ordinal INTEGER, page_or_section TEXT,
    line_start INTEGER, line_end INTEGER, char_start INTEGER, char_end INTEGER,
    text_preview TEXT);
"""


class Datastore(Protocol):
    def seed_users(self) -> None: ...
    def get_user_by_email(self, email: str) -> dict | None: ...
    def add_document(self, **kwargs) -> None: ...
    def add_chunks_meta(self, chunks: list[Chunk]) -> None: ...


class SQLiteDatastore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def seed_users(self) -> None:
        # Phase 0: 2 seeded users, plaintext dev passwords (RBAC/hashing = Phase 1).
        self._conn.executemany(
            "INSERT OR IGNORE INTO users(email, role, password) VALUES (?,?,?)",
            [("admin@genacademy.local", "admin", "admin"),
             ("member@genacademy.local", "member", "member")])
        self._conn.commit()

    def get_user_by_email(self, email: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(row) if row else None

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
        row = self._conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def add_chunks_meta(self, chunks: list[Chunk]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks_meta(id,doc_id,ordinal,page_or_section,line_start,"
            "line_end,char_start,char_end,text_preview) VALUES (?,?,?,?,?,?,?,?,?)",
            [(c.chunk_id, c.doc_id, c.ordinal, c.citation.page_or_section, c.citation.line_start,
              c.citation.line_end, c.citation.char_start, c.citation.char_end, c.text[:200])
             for c in chunks])
        self._conn.commit()

    def get_chunks_for_doc(self, doc_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM chunks_meta WHERE doc_id=? ORDER BY ordinal", (doc_id,)).fetchall()
        return [dict(r) for r in rows]
