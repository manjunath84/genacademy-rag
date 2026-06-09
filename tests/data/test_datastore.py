import sqlite3
from concurrent.futures import ThreadPoolExecutor

from genacademy_rag.core.security import hash_password, is_bcrypt_hash, verify_password
from genacademy_rag.core.types import Chunk, Citation
from genacademy_rag.data.datastore import SQLiteDatastore


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


def test_seed_users_and_lookup(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    ds.seed_users()
    admin = ds.get_user_by_email("admin@genacademy.local")
    member = ds.get_user_by_email("member@genacademy.local")
    assert admin is not None and admin["role"] == "admin"
    assert member is not None and member["role"] == "member"


def test_record_document_and_chunks(tmp_path):
    ds = SQLiteDatastore(tmp_path / "t.sqlite")
    ds.add_document(
        doc_id="d1",
        title="README.md",
        source_type="github",
        repo="r",
        file_path="README.md",
        commit_hash="abc123",
        n_chunks=2,
    )
    ds.add_chunks_meta([_chunk(0), _chunk(1)])
    doc = ds.get_document("d1")
    assert doc is not None and doc["commit_hash"] == "abc123" and doc["n_chunks"] == 2
    metas = ds.get_chunks_for_doc("d1")
    assert len(metas) == 2 and metas[0]["line_start"] == 0


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
    revoked = ds.generate_invite(
        role="member", created_by="admin@genacademy.local", expires_at=None
    )
    expired = ds.generate_invite(
        role="member",
        created_by="admin@genacademy.local",
        expires_at="2000-01-01 00:00:00",
    )
    ds.revoke_invite(revoked["id"])
    assert (
        ds.redeem_invite(
            raw_code=active["id"] + ".wrong",
            email="bad@example.com",
            password_hash=hash_password("pw"),
        )
        is None
    )
    assert (
        ds.redeem_invite(
            raw_code=revoked["code"],
            email="revoked@example.com",
            password_hash=hash_password("pw"),
        )
        is None
    )
    assert (
        ds.redeem_invite(
            raw_code=expired["code"],
            email="expired@example.com",
            password_hash=hash_password("pw"),
        )
        is None
    )


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
