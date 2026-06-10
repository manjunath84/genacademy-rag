from types import SimpleNamespace

import pytest

from genacademy_rag.core.types import Chunk, Citation
from genacademy_rag.core.vectorstore import (
    ChromaStore,
    PineconeStore,
    VectorStoreSetupError,
    build_vectorstore,
)


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


# --- PineconeStore (offline: fake client records SDK calls, no network) ---


class _FakeIndex:
    def __init__(self):
        self.vectors: dict[str, dict] = {}   # id -> {"values": [...], "metadata": {...}}
        self.calls: list[tuple] = []

    def upsert(self, *, vectors, namespace):
        self.calls.append(("upsert", [v["id"] for v in vectors], namespace))
        for v in vectors:
            self.vectors[v["id"]] = {"values": v["values"], "metadata": v["metadata"]}

    def query(self, *, vector, top_k, namespace):
        self.calls.append(("query", top_k, namespace))
        # Pinecone cosine score IS similarity. Return ids in insertion order, fake scores.
        matches = [
            SimpleNamespace(id=cid, score=0.9 - 0.1 * i)
            for i, cid in enumerate(self.vectors)
        ][:top_k]
        return SimpleNamespace(matches=matches)

    def fetch(self, *, ids, namespace):
        self.calls.append(("fetch", list(ids), namespace))
        found = {
            cid: SimpleNamespace(metadata=self.vectors[cid]["metadata"])
            for cid in ids
            if cid in self.vectors
        }
        return SimpleNamespace(vectors=found)

    def list(self, *, namespace, prefix=None):
        self.calls.append(("list", prefix, namespace))
        ids = [cid for cid in self.vectors if prefix is None or cid.startswith(prefix)]
        # Yield two pages to prove pagination is followed.
        mid = max(1, len(ids) // 2) if ids else 0
        for page_ids in (ids[:mid], ids[mid:]):
            if page_ids:
                yield SimpleNamespace(vectors=[SimpleNamespace(id=i) for i in page_ids])

    def delete(self, *, ids, namespace):
        self.calls.append(("delete", list(ids), namespace))
        for cid in ids:
            self.vectors.pop(cid, None)


class _FakePinecone:
    def __init__(self, *, existing_indexes=()):
        self.existing = set(existing_indexes)
        self.created: list[dict] = []
        self.index = _FakeIndex()

    def has_index(self, name):
        return name in self.existing

    def create_index(self, *, name, dimension, metric, spec):
        self.created.append({"name": name, "dimension": dimension, "metric": metric, "spec": spec})
        self.existing.add(name)

    def Index(self, name):  # noqa: N802 — matches the SDK method name
        return self.index


def _pinecone_store(client=None, **kwargs):
    client = client or _FakePinecone(existing_indexes=("genacademy-rag",))
    store = PineconeStore(
        api_key="test-key", index_name="genacademy-rag", namespace="serving",
        client=client, **kwargs,
    )
    return store, client


def test_pinecone_creates_missing_index_with_384_cosine():
    client = _FakePinecone()                              # index does not exist yet
    _pinecone_store(client=client)
    assert len(client.created) == 1
    created = client.created[0]
    assert created["name"] == "genacademy-rag"
    assert created["dimension"] == 384                    # all-MiniLM-L6-v2 dim (tech-stack.md)
    assert created["metric"] == "cosine"
    assert created["spec"].cloud == "aws"
    assert created["spec"].region == "us-east-1"


def test_pinecone_does_not_recreate_existing_index():
    store, client = _pinecone_store()
    assert client.created == []
    assert store is not None


def test_pinecone_upsert_strips_none_metadata_and_stores_text_and_ordinal():
    store, client = _pinecone_store()
    cit = Citation(doc_id="d1", title="t.pdf", source_type="pdf")  # repo/file_path/... all None
    chunk = Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0, text="alpha", citation=cit)
    store.upsert([chunk], [[0.1] * 384])

    stored = client.index.vectors["d1::0"]
    assert stored["metadata"]["text"] == "alpha"
    assert stored["metadata"]["ordinal"] == 0
    assert None not in stored["metadata"].values()        # Pinecone rejects null metadata
    assert ("upsert", ["d1::0"], "serving") in client.index.calls


def test_pinecone_query_passes_score_through_as_similarity():
    store, client = _pinecone_store()
    store.upsert([_chunk(0, "alpha"), _chunk(1, "beta")], [[0.1] * 384, [0.2] * 384])

    results = store.query([0.1] * 384, top_k=2)

    assert results == [("d1::0", 0.9), ("d1::1", pytest.approx(0.8))]  # no 1-dist conversion


def test_pinecone_chunk_round_trip_coerces_numeric_metadata_to_int():
    store, client = _pinecone_store()
    store.upsert([_chunk(3, "gamma")], [[0.3] * 384])
    # Pinecone returns JSON numbers as floats; simulate that on the stored metadata.
    meta = client.index.vectors["d1::3"]["metadata"]
    for key in ("ordinal", "line_start", "line_end", "char_start", "char_end"):
        meta[key] = float(meta[key])

    got = store.get_chunk("d1::3")

    assert got.text == "gamma"
    assert got.ordinal == 3 and isinstance(got.ordinal, int)
    assert got.citation.line_start == 3 and isinstance(got.citation.line_start, int)
    assert got.citation.char_end == 4 and isinstance(got.citation.char_end, int)
    assert got.citation.commit_hash == "abc123"


def test_pinecone_get_all_chunks_paginates_and_sorts_by_doc_and_ordinal():
    store, client = _pinecone_store()
    chunks = [_chunk(2, "c"), _chunk(0, "a"), _chunk(1, "b"), _chunk(10, "k")]
    store.upsert(chunks, [[0.1] * 384] * 4)

    got = store.get_all_chunks()

    # Sorted numerically by (doc_id, ordinal) — not lexically, where d1::10 < d1::2.
    assert [c.chunk_id for c in got] == ["d1::0", "d1::1", "d1::2", "d1::10"]


def test_pinecone_upsert_and_fetch_batch_at_100():
    store, client = _pinecone_store()
    chunks = [_chunk(i, f"text {i}") for i in range(101)]

    store.upsert(chunks, [[0.1] * 384] * 101)

    upsert_calls = [c for c in client.index.calls if c[0] == "upsert"]
    assert [len(c[1]) for c in upsert_calls] == [100, 1]
    assert len(client.index.vectors) == 101

    got = store.get_all_chunks()

    fetch_calls = [c for c in client.index.calls if c[0] == "fetch"]
    assert [len(c[1]) for c in fetch_calls] == [100, 1]
    assert len(got) == 101


def test_pinecone_get_all_chunks_empty_namespace_returns_empty_without_fetch():
    store, client = _pinecone_store()

    assert store.get_all_chunks() == []
    # The real SDK rejects fetch(ids=[]) — the empty namespace must never reach fetch.
    assert all(call[0] != "fetch" for call in client.index.calls)


def test_pinecone_get_all_chunks_warns_on_partial_fetch(caplog):
    store, client = _pinecone_store()
    store.upsert([_chunk(0, "a"), _chunk(1, "b")], [[0.1] * 384] * 2)
    original_fetch = client.index.fetch

    def partial_fetch(*, ids, namespace):
        res = original_fetch(ids=ids, namespace=namespace)
        res.vectors.pop("d1::1", None)     # fetch silently omits a listed id
        return res

    client.index.fetch = partial_fetch
    with caplog.at_level("WARNING"):
        got = store.get_all_chunks()

    assert [c.chunk_id for c in got] == ["d1::0"]
    assert any("listed 2 ids but fetched 1" in r.message for r in caplog.records)


def test_pinecone_delete_doc_warns_when_listing_returns_nothing(caplog):
    store, client = _pinecone_store()

    with caplog.at_level("WARNING"):
        store.delete_doc("d-unknown")

    assert any("orphaned" in r.message for r in caplog.records)
    assert all(call[0] != "delete" for call in client.index.calls)


def test_pinecone_foreign_vector_without_metadata_raises_with_context():
    store, client = _pinecone_store()
    client.index.vectors["alien::0"] = {"values": [0.1], "metadata": None}

    with pytest.raises(ValueError, match="alien::0.*serving.*written by something other"):
        store.get_chunk("alien::0")


def test_pinecone_get_chunk_missing_id_raises_clear_keyerror():
    store, _client = _pinecone_store()

    with pytest.raises(KeyError, match="serving.*d1::404"):
        store.get_chunk("d1::404")


def test_pinecone_delete_doc_batches_at_api_cap():
    store, client = _pinecone_store()
    # Populate 1001 ids directly; the API caps delete at 1000 ids per request.
    for i in range(1001):
        client.index.vectors[f"d1::{i}"] = {"values": [], "metadata": {}}

    store.delete_doc("d1")

    delete_calls = [c for c in client.index.calls if c[0] == "delete"]
    assert [len(c[1]) for c in delete_calls] == [1000, 1]
    assert client.index.vectors == {}


def test_pinecone_delete_doc_lists_by_id_prefix_then_deletes():
    store, client = _pinecone_store()
    other_cit = Citation(doc_id="d2", title="o", source_type="pdf")
    other = Chunk(chunk_id="d2::0", doc_id="d2", ordinal=0, text="other", citation=other_cit)
    store.upsert([_chunk(0, "a"), _chunk(1, "b"), other], [[0.1] * 384] * 3)

    store.delete_doc("d1")

    assert ("list", "d1::", "serving") in client.index.calls
    assert sorted(client.index.vectors) == ["d2::0"]      # serverless: no filtered deletes


def test_build_vectorstore_defaults_to_chroma(tmp_path, monkeypatch):
    from dataclasses import replace

    from genacademy_rag.config import Settings

    monkeypatch.delenv("GENACADEMY_VECTORSTORE", raising=False)
    s = replace(Settings.from_env(), chroma_dir=tmp_path / "chroma")
    assert s.vectorstore == "chroma"
    assert isinstance(build_vectorstore(s, collection="serving"), ChromaStore)


def test_build_vectorstore_pinecone_without_key_fails_loudly(monkeypatch, tmp_path):
    from genacademy_rag.config import Settings

    monkeypatch.setenv("GENACADEMY_VECTORSTORE", "pinecone")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    s = Settings.from_env()

    with pytest.raises(VectorStoreSetupError, match="PINECONE_API_KEY"):
        build_vectorstore(s, collection="serving")


def test_build_vectorstore_pinecone_passes_settings_and_namespace(monkeypatch):
    import genacademy_rag.core.vectorstore as vs_module
    from genacademy_rag.config import Settings

    monkeypatch.setenv("GENACADEMY_VECTORSTORE", "pinecone")
    monkeypatch.setenv("PINECONE_API_KEY", "pk-test")
    monkeypatch.setenv("GENACADEMY_PINECONE_INDEX", "custom-index")
    monkeypatch.setenv("GENACADEMY_PINECONE_CLOUD", "gcp")
    monkeypatch.setenv("GENACADEMY_PINECONE_REGION", "us-central1")
    recorded = {}

    class _Recorder:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(vs_module, "PineconeStore", _Recorder)
    build_vectorstore(Settings.from_env(), collection="serving")

    assert recorded == {
        "api_key": "pk-test",
        "index_name": "custom-index",
        "namespace": "serving",
        "dimension": 384,
        "cloud": "gcp",
        "region": "us-central1",
    }


def test_build_vectorstore_rejects_unknown_name():
    from dataclasses import replace

    from genacademy_rag.config import Settings

    s = replace(Settings.from_env(), vectorstore="faiss")
    with pytest.raises(ValueError, match="unknown vectorstore"):
        build_vectorstore(s, collection="serving")
