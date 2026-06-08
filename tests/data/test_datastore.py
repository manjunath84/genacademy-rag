from genacademy_rag.core.types import Chunk, Citation
from genacademy_rag.data.datastore import SQLiteDatastore


def _chunk(i):
    cit = Citation(
        doc_id="d1",
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
        chunk_id=f"d1::{i}",
        doc_id="d1",
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
