"""Run the 15-question eval over the ingested pinned corpus and write the report.
Retrieval eval always runs (protected). LLM-judge runs unless --no-judge or throttling forces
the citation-grounding fallback. Saves raw judge outputs to eval/runs/ for auditability."""
import argparse
import json
import sys
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
    # (row, question, answer, retrieved) for each answerable question — faithfulness is scored in a
    # second pass so the report's scorer label always matches every row (no judge/grounding mix).
    answerable_rows: list[tuple[dict, str, str, list]] = []
    for q in gold:
        retrieved = retriever.retrieve(q.question)
        row = score_question(q, retrieved, k=s.top_k)
        result = qp.answer(q.question)
        row["refused"] = result.refused
        row["refusal_correct"] = result.refused != q.answerable
        if q.answerable:
            answerable_rows.append((row, q.question, result.answer, retrieved))
        else:
            row["faithful"] = None
        per_q.append(row)

    # Pass 2: try the LLM judge; on the FIRST failure, log which question and why, then disable the
    # judge run-wide and re-score EVERY answerable row with citation-grounding (label stays honest).
    judge_used = not args.no_judge
    if judge_used:
        for row, question, answer, retrieved in answerable_rows:
            try:
                j = llm_judge_score(question, answer, retrieved, provider)
                (runs_dir / f"judge_{row['id']}.json").write_text(json.dumps(j, indent=2))
                row["faithful"] = j["faithful"]
            except Exception as e:           # noqa: BLE001
                print(f"[run_eval] judge failed on q={row['id']}: {type(e).__name__}: {e}; "
                      "disabling judge run-wide, falling back to citation-grounding",
                      file=sys.stderr)
                judge_used = False
                break
    if not judge_used:
        for row, _question, answer, retrieved in answerable_rows:
            row["faithful"] = citation_grounding_score(answer, retrieved)

    agg = aggregate(per_q)
    faith_flags = [bool(row["faithful"]) for row, *_ in answerable_rows]
    faith_pct = (sum(faith_flags) / len(faith_flags)) if faith_flags else None
    failures: list[dict] = []  # fill from per_q rows where recall<1 / refusal_correct is False
    md = render_report(agg, per_q, failures, faithfulness_pct=faith_pct, judge_used=judge_used)
    out = Path("eval/REPORT.md")
    out.write_text(md)
    print(f"wrote {out} | recall@k={agg['recall@k']:.2f} precision@k={agg['precision@k']:.2f} "
          f"mrr={agg['mrr']:.2f} judge_used={judge_used}")


if __name__ == "__main__":
    main()
