"""Day-2 gate: deterministic retrieval eval over the ingested pinned corpus. Prints recall@k /
precision@k / MRR. No LLM, no generation key. (Full report + faithfulness = scripts/run_eval.py.)

Run results (2026-06-08, top_k=5, all-MiniLM-L6-v2, HybridRetriever RRF):
  RETRIEVAL EVAL  recall@k=0.67  precision@k=0.22  mrr=0.55  (n=12)
  q1    answerable       recall=1.00 mrr=1.00
  q2    answerable       recall=1.00 mrr=1.00
  q3    answerable       recall=1.00 mrr=1.00
  q4    answerable       recall=1.00 mrr=0.33
  q5    exact_match      recall=0.00 mrr=0.00  # ChunkingBoundary: answer spans chunk boundary
  q6    exact_match      recall=1.00 mrr=1.00
  q7    chunking_stress  recall=0.00 mrr=0.00  # ChunkingBoundary: answer split across chunks
  q8    chunking_stress  recall=1.00 mrr=0.50
  q9    multi_document   recall=0.50 mrr=0.50  # one of two gold spans below top_k=5
  q10   multi_document   recall=0.50 mrr=0.25  # one of two gold spans below top_k=5
  q11   ambiguous        recall=1.00 mrr=1.00
  q12   ambiguous        recall=0.00 mrr=0.00  # ChunkingBoundary: answer split across chunks
"""
import argparse
import json
import time
from pathlib import Path

from genacademy_rag.config import Settings
from genacademy_rag.core.providers import STEmbedder
from genacademy_rag.core.reranker import build_reranker
from genacademy_rag.core.retriever import DEFAULT_CANDIDATE_K, HybridRetriever
from genacademy_rag.core.vectorstore import ChromaStore
from genacademy_rag.eval.gold_schema import load_gold_set
from genacademy_rag.eval.retrieval_eval import (
    aggregate,
    build_retrieval_eval_payload,
    score_question,
)

GOLD = "src/genacademy_rag/eval/gold/gold_set.yaml"


def _config_snapshot(settings: Settings) -> dict:
    return {
        "collection": "eval",
        "top_k": settings.top_k,
        "candidate_k": DEFAULT_CANDIDATE_K,
        "embed_model": settings.embed_model,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_model": settings.rerank_model,
        "rerank_pool": settings.rerank_pool,
        "rerank_device": settings.rerank_device or "",
        "rerank_batch_size": settings.rerank_batch_size,
        "rerank_local_files_only": settings.rerank_local_files_only,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    s = Settings.from_env()
    store = ChromaStore(persist_dir=s.chroma_dir, collection="eval")
    chunks = store.get_all_chunks()
    # Embeddings only — no generate() — so this runs with zero provider key.
    embedder = STEmbedder(s.embed_model)
    retriever = HybridRetriever(
        store=store,
        provider=embedder,
        all_chunks=chunks,
        top_k=s.top_k,
        candidate_k=DEFAULT_CANDIDATE_K,
        reranker=build_reranker(s),
        rerank_pool=s.rerank_pool,
    )
    scores = []
    for q in load_gold_set(GOLD):
        started = time.perf_counter()
        retrieved = retriever.retrieve(q.question)
        retrieval_ms = (time.perf_counter() - started) * 1000
        row = score_question(q, retrieved, k=s.top_k)
        row["question"] = q.question
        row["retrieval_ms"] = round(retrieval_ms, 3)
        row["max_cosine"] = max((r.score for r in retrieved), default=0.0)
        row["cosine_fallback_answerable_at_0_2"] = row["max_cosine"] >= 0.2
        row["retrieved"] = [
            {
                "rank": i + 1,
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.doc_id,
                "title": r.chunk.citation.title,
                "score": r.score,
                "repo": r.chunk.citation.repo,
                "file_path": r.chunk.citation.file_path,
                "commit_hash": r.chunk.citation.commit_hash,
                "line_start": r.chunk.citation.line_start,
                "line_end": r.chunk.citation.line_end,
                "page_or_section": r.chunk.citation.page_or_section,
            }
            for i, r in enumerate(retrieved)
        ]
        scores.append(row)
    agg = aggregate(scores)
    print(
        f"RETRIEVAL EVAL  recall@k={agg['recall@k']:.2f}  "
        f"precision@k={agg['precision@k']:.2f}  "
        f"mrr={agg['mrr']:.2f}  (n={agg['n_retrieval_questions']})"
    )
    for row in scores:
        if row["recall"] is not None:
            print(
                f"  {row['id']:<5} {row['category']:<16} "
                f"recall={row['recall']:.2f} mrr={row['mrr']:.2f}"
            )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = build_retrieval_eval_payload(
            metrics=agg,
            rows=scores,
            config=_config_snapshot(s),
        )
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
