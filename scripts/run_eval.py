"""Run the 15-question eval over the ingested pinned corpus and write the report.
Retrieval eval always runs (protected). LLM-judge runs unless --no-judge or throttling forces
the citation-grounding fallback. Saves raw judge outputs to eval/runs/ for auditability."""
import argparse
import json
from pathlib import Path

from genacademy_rag.config import Settings
from genacademy_rag.core.pipeline import QueryPipeline
from genacademy_rag.core.providers import build_provider
from genacademy_rag.core.retriever import HybridRetriever
from genacademy_rag.core.vectorstore import ChromaStore
from genacademy_rag.eval.faithfulness_eval import citation_grounding_score, llm_judge_score
from genacademy_rag.eval.gold_schema import load_gold_set
from genacademy_rag.eval.report import render_report
from genacademy_rag.eval.retrieval_eval import aggregate, score_question

GOLD = "src/genacademy_rag/eval/gold/gold_set.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    s = Settings.from_env()
    provider = build_provider(s)
    store = ChromaStore(persist_dir=s.chroma_dir, collection="eval")
    chunks = store.get_all_chunks()                  # public accessor, not store._col
    retriever = HybridRetriever(store=store, provider=provider, all_chunks=chunks, top_k=s.top_k)
    qp = QueryPipeline(retriever=retriever, provider=provider)
    gold = load_gold_set(GOLD)

    runs_dir = Path("eval/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    per_q: list[dict] = []
    faith_flags: list[bool] = []
    judge_used = not args.no_judge
    for q in gold:
        retrieved = retriever.retrieve(q.question)
        row = score_question(q, retrieved, k=s.top_k)
        result = qp.answer(q.question)
        row["refused"] = result.refused
        row["refusal_correct"] = result.refused != q.answerable
        if q.answerable:
            if judge_used:
                try:
                    j = llm_judge_score(q.question, result.answer, retrieved, provider)
                    (runs_dir / f"judge_{q.id}.json").write_text(json.dumps(j, indent=2))
                    row["faithful"] = j["faithful"]
                except Exception:           # noqa: BLE001
                    # Deliberate: disable the judge run-wide on first failure (throttling/parse)
                    # and fall back to citation-grounding for ALL questions. A report mixing two
                    # faithfulness scorers is incoherent; one labeled scorer is the honest output.
                    judge_used = False
            if not judge_used:
                row["faithful"] = citation_grounding_score(result.answer, retrieved)
            faith_flags.append(bool(row["faithful"]))
        else:
            row["faithful"] = None
        per_q.append(row)

    agg = aggregate(per_q)
    faith_pct = (sum(faith_flags) / len(faith_flags)) if faith_flags else None
    failures: list[dict] = []  # fill from per_q rows where recall<1 / refusal_correct is False
    md = render_report(agg, per_q, failures, faithfulness_pct=faith_pct, judge_used=judge_used)
    out = Path("eval/REPORT.md")
    out.write_text(md)
    print(f"wrote {out} | recall@k={agg['recall@k']:.2f} precision@k={agg['precision@k']:.2f} "
          f"mrr={agg['mrr']:.2f} judge_used={judge_used}")


if __name__ == "__main__":
    main()
