"""Retriever seam. HybridRetriever = dense (VectorStore) + sparse (rank-bm25) fused via RRF.
Phase-2 swap target: + cross-encoder rerank."""
from __future__ import annotations

import re
from typing import Protocol

from rank_bm25 import BM25Okapi

from genacademy_rag.core.types import Chunk, RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedChunk]: ...


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score = sum over lists of 1/(k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    def __init__(self, *, store, provider, all_chunks: list[Chunk], top_k: int = 5,
                 candidate_k: int = 20, rrf_k: int = 60):
        self._store = store
        self._provider = provider
        self._top_k = top_k
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._chunks_by_id = {c.chunk_id: c for c in all_chunks}
        self._ids = [c.chunk_id for c in all_chunks]
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in all_chunks])

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        qvec = self._provider.embed([query])[0]
        dense_hits = self._store.query(qvec, top_k=self._candidate_k)   # list[(id, cosine_sim)]
        dense_ids = [cid for cid, _ in dense_hits]
        sim_by_id = {cid: sim for cid, sim in dense_hits}
        scores = self._bm25.get_scores(_tokenize(query))
        bm25_order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        sparse_ids = [self._ids[i] for i in bm25_order][: self._candidate_k]
        fused = rrf_fuse([dense_ids, sparse_ids], k=self._rrf_k)        # ranking signal
        ranked = sorted(fused, key=fused.get, reverse=True)[:self._top_k]
        # score = cosine similarity (the grader's confidence signal); 0.0 for BM25-only hits.
        # RRF decides ORDER; cosine sim is carried separately so the grader fallback is meaningful.
        return [RetrievedChunk(chunk=self._chunks_by_id[cid], score=sim_by_id.get(cid, 0.0))
                for cid in ranked]
