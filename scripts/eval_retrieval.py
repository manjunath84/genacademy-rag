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
from genacademy_rag.config import Settings
from genacademy_rag.core.providers import STEmbedder
from genacademy_rag.core.retriever import HybridRetriever
from genacademy_rag.core.vectorstore import ChromaStore
from genacademy_rag.eval.gold_schema import load_gold_set
from genacademy_rag.eval.retrieval_eval import aggregate, score_question

GOLD = "src/genacademy_rag/eval/gold/gold_set.yaml"


def main():
    s = Settings.from_env()
    store = ChromaStore(persist_dir=s.chroma_dir, collection="eval")
    chunks = store.get_all_chunks()
    # Embeddings only — no generate() — so this runs with zero provider key.
    embedder = STEmbedder(s.embed_model)
    retriever = HybridRetriever(store=store, provider=embedder, all_chunks=chunks, top_k=s.top_k)
    scores = [score_question(q, retriever.retrieve(q.question), k=s.top_k)
              for q in load_gold_set(GOLD)]
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


if __name__ == "__main__":
    main()
