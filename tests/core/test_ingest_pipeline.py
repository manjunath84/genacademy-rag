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
