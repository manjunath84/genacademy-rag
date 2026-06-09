"""Retriever seam. HybridRetriever = dense (VectorStore) + sparse (rank-bm25) fused via RRF.
Phase-2 swap target: + cross-encoder rerank."""
from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rank_bm25 import BM25Okapi

from genacademy_rag.core.types import Chunk, RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedChunk]: ...


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score = sum over lists of 1/(k + rank + 1), rank 0-indexed."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return scores


@dataclass(frozen=True)
class _Index:
    ids: tuple[str, ...]
    chunks_by_id: dict[str, Chunk]
    bm25: BM25Okapi | None


def _build_index(chunks: list[Chunk]) -> _Index:
    chunk_list = list(chunks)
    bm25 = BM25Okapi([_tokenize(c.text) for c in chunk_list]) if chunk_list else None
    return _Index(
        ids=tuple(c.chunk_id for c in chunk_list),
        chunks_by_id={c.chunk_id: c for c in chunk_list},
        bm25=bm25,
    )


class HybridRetriever:
    def __init__(self, *, store, provider, all_chunks: list[Chunk], top_k: int = 5,
                 candidate_k: int = 20, rrf_k: int = 60):
        self._store = store
        self._provider = provider
        self._top_k = top_k
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._corpus_lock = threading.Lock()
        self._index = _build_index(all_chunks)

    def _swap_index_unlocked(self, all_chunks: list[Chunk]) -> None:
        self._index = _build_index(all_chunks)

    def reindex(self, all_chunks: list[Chunk]) -> None:
        with self._corpus_lock:
            self._swap_index_unlocked(all_chunks)

    def mutate_corpus(self, mutation: Callable[[], list[Chunk]]) -> None:
        with self._corpus_lock:
            self._swap_index_unlocked(mutation())

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        with self._corpus_lock:
            index = self._index
            qvec = self._provider.embed([query])[0]
            dense_hits = self._store.query(qvec, top_k=self._candidate_k)   # list[(id, cosine_sim)]
            dense_ids = [cid for cid, _ in dense_hits if cid in index.chunks_by_id]
            sim_by_id = {cid: sim for cid, sim in dense_hits if cid in index.chunks_by_id}
            if index.bm25 is None:
                sparse_ids: list[str] = []
            else:
                scores = index.bm25.get_scores(_tokenize(query))
                bm25_order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
                sparse_ids = [index.ids[i] for i in bm25_order][: self._candidate_k]
            fused = rrf_fuse([dense_ids, sparse_ids], k=self._rrf_k)        # ranking signal
            ranked = sorted(fused, key=fused.get, reverse=True)[:self._top_k]
            # score = cosine similarity (the grader's confidence signal); 0.0 for BM25-only hits.
            # RRF decides ORDER; cosine sim is carried separately for the grader fallback.
            return [
                RetrievedChunk(chunk=index.chunks_by_id[cid], score=sim_by_id.get(cid, 0.0))
                for cid in ranked
            ]
