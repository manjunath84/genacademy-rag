import pytest

from genacademy_rag.core.types import Chunk, Citation
from genacademy_rag.core.vectorstore import ChromaStore


def _chunk(i, text):
    cit = Citation(doc_id="d1", title="README.md", source_type="github",
                   repo="r", file_path="README.md", commit_hash="abc123",
                   line_start=i, line_end=i, char_start=i, char_end=i + 1)
    return Chunk(chunk_id=f"d1::{i}", doc_id="d1", ordinal=i, text=text, citation=cit)


def test_upsert_then_query_returns_nearest_chunk_ids(tmp_path, fake_provider):
    store = ChromaStore(persist_dir=tmp_path / "chroma", collection="tcol")
    chunks = [_chunk(0, "retrieval augmented generation"), _chunk(1, "banana bread recipe")]
    embs = fake_provider.embed([c.text for c in chunks])
    store.upsert(chunks, embs)
    qvec = fake_provider.embed(["retrieval augmented generation"])[0]
    results = store.query(qvec, top_k=2)               # list[(chunk_id, cosine_similarity)]
    ids = [cid for cid, _ in results]
    assert ids[0] == "d1::0"  # exact-text match ranks first under the deterministic fake embed
    assert set(ids) == {"d1::0", "d1::1"}
    assert results[0][1] >= results[1][1]              # similarity descending
    assert results[0][1] == pytest.approx(1.0, abs=1e-3)  # query == chunk text -> sim ~1.0


def test_get_chunk_round_trips_citation(tmp_path, fake_provider):
    store = ChromaStore(persist_dir=tmp_path / "chroma", collection="tcol")
    chunks = [_chunk(0, "alpha")]
    store.upsert(chunks, fake_provider.embed(["alpha"]))
    got = store.get_chunk("d1::0")
    assert got.text == "alpha"
    assert got.citation.commit_hash == "abc123"
    assert got.citation.line_start == 0


def test_get_all_chunks_returns_every_upserted_chunk(tmp_path, fake_provider):
    store = ChromaStore(persist_dir=tmp_path / "chroma", collection="tcol")
    chunks = [_chunk(0, "alpha"), _chunk(1, "beta")]
    store.upsert(chunks, fake_provider.embed([c.text for c in chunks]))
    got = {c.chunk_id for c in store.get_all_chunks()}
    assert got == {"d1::0", "d1::1"}


def test_chroma_delete_doc_removes_only_matching_doc(tmp_path, fake_provider):
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
