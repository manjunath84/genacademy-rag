"""Render the eval report markdown: a scores table (retrieval columns ALWAYS; faithfulness % from
whichever scorer survived) + a failure-analysis table (Symptom → Cause[taxonomy] → Fix)."""
from __future__ import annotations

TAXONOMY = ["ChunkingBoundary", "RetrievalRecallFailure", "FaithfulnessHallucination",
            "RefusalFalsePositive", "RefusalFalseNegative", "TopKTooSmall"]


def render_report(agg: dict, per_q: list[dict], failures: list[dict],
                  *, faithfulness_pct: float | None, judge_used: bool) -> str:
    lines = ["# GenAcademy RAG — Evaluation Report", ""]
    lines += ["## Scores", "",
              "| Metric | Value |", "|---|---|",
              f"| Retrieval questions | {agg['n_retrieval_questions']} |",
              f"| recall@k | {agg['recall@k']:.2f} |",
              f"| precision@k | {agg['precision@k']:.2f} |",
              f"| MRR | {agg['mrr']:.2f} |"]
    refusal_correct = [q for q in per_q if "refusal_correct" in q]
    if refusal_correct:
        rate = sum(q["refusal_correct"] for q in refusal_correct) / len(refusal_correct)
        lines.append(f"| refusal correctness | {rate:.2f} |")
    if faithfulness_pct is not None:
        src = "LLM-judge" if judge_used else "citation-grounding fallback"
        lines.append(f"| faithfulness % ({src}) | {faithfulness_pct * 100:.0f}% |")
    lines += ["", "## Per-question", "",
              "| id | category | recall | precision | mrr | refused | faithful |",
              "|---|---|---|---|---|---|---|"]
    for q in per_q:
        def fmt(v):
            if v is None:
                return "—"
            return f"{v:.2f}" if isinstance(v, float) else str(v)
        lines.append(f"| {q['id']} | {q['category']} | {fmt(q.get('recall'))} | "
                     f"{fmt(q.get('precision'))} | {fmt(q.get('mrr'))} | "
                     f"{q.get('refused', '—')} | {fmt(q.get('faithful'))} |")
    lines += ["", "## Failure analysis", "",
              "| Symptom | Cause | Fix | Question |", "|---|---|---|---|"]
    for f in failures:
        lines.append(f"| {f['symptom']} | {f['cause']} | {f['fix']} | {f.get('qid', '—')} |")
    return "\n".join(lines) + "\n"
