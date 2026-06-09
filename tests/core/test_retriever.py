from genacademy_rag.core.retriever import HybridRetriever, rrf_fuse
from genacademy_rag.core.types import Chunk, Citation


def _chunk(i, text):
    cit = Citation(doc_id="d1", title="t", source_type="github", repo="r",
                   file_path="f", commit_hash="abc123", line_start=i, line_end=i)
    return Chunk(chunk_id=f"d1::{i}", doc_id="d1", ordinal=i, text=text, citation=cit)


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
