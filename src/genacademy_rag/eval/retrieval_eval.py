"""Deterministic retrieval eval (handout-mandatory). recall@k / precision@k / MRR over gold spans.
A retrieved chunk counts only if it overlaps a gold span AND its commit_hash matches the gold's —
so production content (different/missing commit_hash) can never satisfy a gold marker."""
from __future__ import annotations

from genacademy_rag.core.types import Chunk
from genacademy_rag.eval.gold_schema import GoldQuestion, GoldSpan


def chunk_matches_span(chunk: Chunk, span: GoldSpan) -> bool:
    c = chunk.citation
    if c.repo != span.repo or c.file_path != span.file_path:
        return False
    if c.commit_hash != span.commit_hash:   # provenance gate: no production leak into gold
        return False
    if span.section is not None:
        return (c.page_or_section or "") == span.section
    if span.line_start is None or c.line_start is None:
        return True
    return not (c.line_end < span.line_start or c.line_start > span.line_end)  # overlap


def score_question(q: GoldQuestion, retrieved: list, k: int) -> dict:
    if not q.answerable:
        return {"id": q.id, "category": q.category, "recall": None, "precision": None, "mrr": None}
    topk = retrieved[:k]
    hits = [any(chunk_matches_span(r.chunk, s) for s in q.gold) for r in topk]
    gold_found = sum(1 for s in q.gold if any(chunk_matches_span(r.chunk, s) for r in topk))
    recall = gold_found / len(q.gold) if q.gold else 0.0
    precision = sum(hits) / k
    first = next((i for i, h in enumerate(hits) if h), None)
    mrr = 1.0 / (first + 1) if first is not None else 0.0
    return {"id": q.id, "category": q.category,
            "recall": recall, "precision": precision, "mrr": mrr}


def aggregate(scores: list[dict]) -> dict:
    retr = [s for s in scores if s["recall"] is not None]
    n = len(retr)

    def mean(key: str) -> float:
        return (sum(s[key] for s in retr) / n) if n else 0.0

    return {"n_retrieval_questions": n, "recall@k": mean("recall"),
            "precision@k": mean("precision"), "mrr": mean("mrr")}
