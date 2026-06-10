from genacademy_rag.core.chunker import FixedSizeChunker
from genacademy_rag.core.pipeline import IngestPipeline
from genacademy_rag.core.types import Document
from genacademy_rag.core.vectorstore import ChromaStore


def test_ingest_chunks_embeds_stores_and_records(tmp_path, fake_provider):
    store = ChromaStore(persist_dir=tmp_path / "chroma", collection="tcol")

    class _DS:
        def __init__(self):
            self.docs, self.chunks = [], []

        def add_document(self, **kw):
            self.docs.append(kw)

        def add_chunks_meta(self, chunks):
            self.chunks.extend(chunks)

    ds = _DS()
    pipe = IngestPipeline(
        chunker=FixedSizeChunker(chunk_size=50, overlap=10),
        provider=fake_provider,
        store=store,
        datastore=ds,
    )
    doc = Document(
        doc_id="d1",
        title="README.md",
        source_type="github",
        text="x" * 200,
        repo="r",
        file_path="README.md",
        commit_hash="abc123",
    )
    n = pipe.ingest([doc])
    assert n > 1                          # multiple chunks produced
    assert ds.docs[0]["n_chunks"] == n
    assert ds.docs[0]["commit_hash"] == "abc123"
    # stored & queryable
    qvec = fake_provider.embed(["x" * 50])[0]
    assert store.query(qvec, top_k=1)


def test_ingest_pipeline_records_upload_provenance(fake_provider):
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


def test_prepared_ingest_embeds_before_persistence(fake_provider):
    calls = []

    class Store:
        def upsert(self, chunks, embeddings):
            calls.append(("upsert", [c.chunk_id for c in chunks], len(embeddings)))

    class Datastore:
        def add_document(self, **kwargs):
            calls.append(("add_document", kwargs["doc_id"], kwargs["n_chunks"]))

        def add_chunks_meta(self, chunks):
            calls.append(("add_chunks_meta", [c.chunk_id for c in chunks]))

    doc = Document(
        doc_id="pdf/abc",
        title="notes.pdf",
        source_type="pdf",
        text="Gen Academy notes about retrieval.",
        filename="notes.pdf",
        uploaded_by="admin@genacademy.local",
    )
    pipe = IngestPipeline(
        chunker=FixedSizeChunker(chunk_size=50, overlap=5),
        provider=fake_provider,
        store=Store(),
        datastore=Datastore(),
    )

    prepared = pipe.prepare([doc])

    assert calls == []
    assert pipe.commit(prepared) == 1
    # Vector store FIRST: if a remote upsert fails, no SQLite row exists, so the admin
    # list never shows a document that is not actually searchable.
    assert calls == [
        ("upsert", ["pdf/abc::0"], 1),
        ("add_document", "pdf/abc", 1),
        ("add_chunks_meta", ["pdf/abc::0"]),
    ]


def test_commit_rolls_back_vectors_when_ledger_write_fails(fake_provider):
    """Upsert-first ordering means a ledger failure would orphan vectors with no admin
    row — and the reindex filter keeps ledger-less chunks (eval seeds), so orphans
    would resurface. The compensating delete_doc prevents that (codex review P2)."""
    calls = []

    class Store:
        def upsert(self, chunks, embeddings):
            calls.append(("upsert", [c.chunk_id for c in chunks]))

        def delete_doc(self, doc_id):
            calls.append(("delete_doc", doc_id))

    class FailingDatastore:
        def add_document(self, **kwargs):
            raise RuntimeError("sqlite locked")

        def add_chunks_meta(self, chunks):
            raise AssertionError("unreachable")

    doc = Document(
        doc_id="pdf/abc",
        title="notes.pdf",
        source_type="pdf",
        text="Gen Academy notes about retrieval.",
        filename="notes.pdf",
        uploaded_by="admin@genacademy.local",
    )
    pipe = IngestPipeline(
        chunker=FixedSizeChunker(chunk_size=50, overlap=5),
        provider=fake_provider,
        store=Store(),
        datastore=FailingDatastore(),
    )

    try:
        pipe.commit(pipe.prepare([doc]))
    except RuntimeError as exc:
        assert "sqlite locked" in str(exc)   # original error surfaces, not the rollback
    else:
        raise AssertionError("expected RuntimeError")

    assert calls == [("upsert", ["pdf/abc::0"]), ("delete_doc", "pdf/abc")]
