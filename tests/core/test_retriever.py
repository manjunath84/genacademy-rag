import threading
import time

from genacademy_rag.core.retriever import HybridRetriever, rrf_fuse
from genacademy_rag.core.types import Chunk, Citation


def _chunk(i, text):
    cit = Citation(doc_id="d1", title="t", source_type="github", repo="r",
                   file_path="f", commit_hash="abc123", line_start=i, line_end=i)
    return Chunk(chunk_id=f"d1::{i}", doc_id="d1", ordinal=i, text=text, citation=cit)


class _DenseStore:
    def __init__(self, hits):
        self.hits = list(hits)

    def query(self, qvec, top_k):
        return self.hits[:top_k]


class _FakeReranker:
    def __init__(self, scores):
        self.scores = dict(scores)
        self.calls = []

    def rerank(self, query, chunks):
        self.calls.append([chunk.chunk_id for chunk in chunks])
        scored = [(chunk, self.scores.get(chunk.chunk_id, 0.0)) for chunk in chunks]
        return sorted(scored, key=lambda item: item[1], reverse=True)


def test_rrf_rewards_items_ranked_high_in_both_lists():
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    fused = rrf_fuse([dense, sparse], k=60)
    # "a" (ranks 0,1) and "b" (ranks 1,0) outrank "c"/"d" (appear once).
    top2 = sorted(fused, key=fused.get, reverse=True)[:2]
    assert set(top2) == {"a", "b"}


def test_hybrid_retriever_surfaces_exact_keyword_via_bm25(fake_provider):
    # A rare proper noun the dense fake-embed would scatter; BM25 must catch it.
    chunks = [
        _chunk(0, "general notes about retrieval and embeddings"),
        _chunk(1, "the QLoRA technique quantizes weights for finetuning"),
        _chunk(2, "more general notes about vector databases"),
    ]

    class _Store:
        def __init__(self, chunks):
            self._by_id = {c.chunk_id: c for c in chunks}
            self._embs = {c.chunk_id: fake_provider.embed([c.text])[0] for c in chunks}

        def query(self, qvec, top_k):
            import math
            def cos(a, b):
                num = sum(x * y for x, y in zip(a, b, strict=False))
                da = math.sqrt(sum(x * x for x in a))
                db = math.sqrt(sum(y * y for y in b))
                return num / (da * db + 1e-9)
            ranked = sorted(self._embs, key=lambda cid: cos(qvec, self._embs[cid]), reverse=True)
            return [(cid, cos(qvec, self._embs[cid])) for cid in ranked[:top_k]]  # (id, cosine_sim)

        def get_chunk(self, cid):
            return self._by_id[cid]

    retr = HybridRetriever(store=_Store(chunks), provider=fake_provider, all_chunks=chunks, top_k=2)
    results = retr.retrieve("QLoRA")
    assert any("QLoRA" in r.chunk.text for r in results)


def test_reindex_makes_new_chunks_searchable(fake_provider):
    c0 = _chunk(0, "original chunk about embeddings")

    class _Store:
        def __init__(self): self.chunks = [c0]
        def query(self, qvec, top_k): return [(c.chunk_id, 0.5) for c in self.chunks][:top_k]
        def get_chunk(self, cid): return next(c for c in self.chunks if c.chunk_id == cid)

    store = _Store()
    retr = HybridRetriever(store=store, provider=fake_provider, all_chunks=[c0], top_k=5)
    new = _chunk(1, "uploaded chunk about Pinecone")
    store.chunks.append(new)
    retr.reindex(store.chunks)
    assert any("Pinecone" in r.chunk.text for r in retr.retrieve("Pinecone"))


def test_retrieved_score_is_cosine_similarity_not_rrf(fake_provider):
    # Regression guard: score must be the cosine sim (usable by the grader's threshold fallback),
    # NOT the tiny RRF score (~0.03) that would make the cosine fallback refuse everything.
    chunk = _chunk(0, "retrieval augmented generation")

    class _Store:
        def query(self, qvec, top_k):
            return [("d1::0", 0.91)]      # high cosine similarity

        def get_chunk(self, cid):
            return chunk

    retr = HybridRetriever(store=_Store(), provider=fake_provider, all_chunks=[chunk], top_k=1)
    [r] = retr.retrieve("retrieval augmented generation")
    assert r.score == 0.91               # cosine sim carried through, not 2/61


def test_rerank_disabled_preserves_rrf_top_k(fake_provider):
    chunks = [
        _chunk(0, "common alpha"),
        _chunk(1, "common beta"),
        _chunk(2, "semantic rescue candidate"),
        _chunk(3, "common delta"),
    ]
    store = _DenseStore([("d1::0", 0.90), ("d1::1", 0.80), ("d1::2", 0.70)])
    retr = HybridRetriever(
        store=store,
        provider=fake_provider,
        all_chunks=chunks,
        top_k=2,
        candidate_k=3,
    )

    assert [r.chunk.chunk_id for r in retr.retrieve("common")] == ["d1::0", "d1::1"]


def test_enabled_rerank_sees_full_fused_union_and_can_rescue_below_top_k(fake_provider):
    chunks = [
        _chunk(0, "common alpha"),
        _chunk(1, "common beta"),
        _chunk(2, "semantic rescue candidate"),
        _chunk(3, "common delta"),
    ]
    store = _DenseStore([("d1::0", 0.90), ("d1::1", 0.80), ("d1::2", 0.70)])
    reranker = _FakeReranker({"d1::2": 9.0, "d1::0": 1.0, "d1::1": 0.5, "d1::3": 0.1})
    retr = HybridRetriever(
        store=store,
        provider=fake_provider,
        all_chunks=chunks,
        top_k=2,
        candidate_k=3,
        reranker=reranker,
    )

    results = retr.retrieve("common")

    assert reranker.calls == [["d1::0", "d1::1", "d1::2", "d1::3"]]
    assert [r.chunk.chunk_id for r in results] == ["d1::2", "d1::0"]
    assert [r.score for r in results] == [0.70, 0.90]


def test_rerank_pool_truncates_by_rrf_rank_before_rerank(fake_provider):
    chunks = [
        _chunk(0, "common alpha"),
        _chunk(1, "common beta"),
        _chunk(2, "semantic rescue candidate"),
        _chunk(3, "common delta"),
    ]
    store = _DenseStore([("d1::0", 0.90), ("d1::1", 0.80), ("d1::2", 0.70)])
    reranker = _FakeReranker({"d1::2": 9.0, "d1::0": 1.0, "d1::1": 0.5, "d1::3": 0.1})
    retr = HybridRetriever(
        store=store,
        provider=fake_provider,
        all_chunks=chunks,
        top_k=2,
        candidate_k=3,
        reranker=reranker,
        rerank_pool=2,
    )

    results = retr.retrieve("common")

    assert reranker.calls == [["d1::0", "d1::1"]]
    assert [r.chunk.chunk_id for r in results] == ["d1::0", "d1::1"]


def test_rerank_ties_preserve_rrf_order_and_repeat_deterministically(fake_provider):
    chunks = [
        _chunk(0, "common alpha"),
        _chunk(1, "common beta"),
        _chunk(2, "semantic only"),
        _chunk(3, "common delta"),
    ]
    store = _DenseStore([("d1::0", 0.90), ("d1::1", 0.80), ("d1::2", 0.70)])
    reranker = _FakeReranker({"d1::0": 1.0, "d1::1": 1.0, "d1::2": 1.0, "d1::3": 1.0})
    retr = HybridRetriever(
        store=store,
        provider=fake_provider,
        all_chunks=chunks,
        top_k=4,
        candidate_k=3,
        reranker=reranker,
    )

    first = [r.chunk.chunk_id for r in retr.retrieve("common")]
    second = [r.chunk.chunk_id for r in retr.retrieve("common")]

    assert first == ["d1::0", "d1::1", "d1::2", "d1::3"]
    assert second == first


def test_reranker_score_never_overwrites_retrieved_cosine_score(fake_provider):
    chunk = _chunk(0, "retrieval augmented generation")
    reranker = _FakeReranker({"d1::0": 999.0})
    retr = HybridRetriever(
        store=_DenseStore([("d1::0", 0.42)]),
        provider=fake_provider,
        all_chunks=[chunk],
        top_k=1,
        reranker=reranker,
    )

    [result] = retr.retrieve("retrieval augmented generation")

    assert result.score == 0.42


def test_reranked_bm25_only_hit_keeps_zero_cosine_score(fake_provider):
    dense = _chunk(0, "dense semantic neighbor")
    bm25_only = _chunk(1, "QLoRA exact keyword")
    filler = _chunk(2, "unrelated filler text")
    reranker = _FakeReranker({"d1::1": 10.0, "d1::0": 1.0})
    retr = HybridRetriever(
        store=_DenseStore([("d1::0", 0.88)]),
        provider=fake_provider,
        all_chunks=[dense, bm25_only, filler],
        top_k=1,
        candidate_k=1,
        reranker=reranker,
    )

    [result] = retr.retrieve("QLoRA")

    assert result.chunk.chunk_id == "d1::1"
    assert result.score == 0.0


def test_reindex_uses_single_snapshot_not_torn_fields(fake_provider):
    old = _chunk(0, "old retrieval text")
    new = _chunk(1, "new Pinecone text")

    class _Store:
        def __init__(self):
            self.chunks = [old]

        def query(self, qvec, top_k):
            return [(c.chunk_id, 0.8) for c in self.chunks]

    store = _Store()
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

    class _Store:
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

    store = _Store()
    retr = HybridRetriever(store=store, provider=fake_provider, all_chunks=store.get_all_chunks(),
                           top_k=5)
    first_results = []

    def retrieve_before_delete():
        first_results.extend(retr.retrieve("QLoRA"))

    thread = threading.Thread(target=retrieve_before_delete)
    thread.start()
    assert query_entered.wait(timeout=2)
    mutation_done = threading.Event()

    def mutate():
        retr.mutate_corpus(lambda: (store.delete_doc("d1"), store.get_all_chunks())[1])
        mutation_done.set()

    mutation_thread = threading.Thread(target=mutate)
    mutation_thread.start()
    time.sleep(0.05)
    assert not mutation_done.is_set()
    release_query.set()
    thread.join(timeout=2)
    mutation_thread.join(timeout=2)
    assert mutation_done.is_set()
    after = retr.retrieve("QLoRA")
    assert all(r.chunk.chunk_id != "d1::1" for r in after)
