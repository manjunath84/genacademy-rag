from genacademy_rag.eval.report import render_report


def test_report_has_scores_table_and_failure_table():
    agg = {"n_retrieval_questions": 12, "recall@k": 0.83, "precision@k": 0.31, "mrr": 0.71}
    per_q = [
        {"id": "q1", "category": "answerable", "recall": 1.0, "precision": 0.2, "mrr": 1.0,
         "refused": False, "refusal_correct": True, "faithful": True},
        {"id": "q13", "category": "unanswerable", "recall": None, "precision": None, "mrr": None,
         "refused": True, "refusal_correct": True, "faithful": None},
    ]
    failures = [{"symptom": "missed gold chunk", "cause": "RetrievalRecallFailure",
                 "fix": "raise candidate_k / inspect BM25 tokenization", "qid": "q7"}]
    md = render_report(agg, per_q, failures, faithfulness_pct=0.92, judge_used=True)
    assert "recall@k" in md and "0.83" in md
    assert "Symptom" in md and "Cause" in md and "Fix" in md      # FIX column required
    assert "RetrievalRecallFailure" in md
    assert "refusal" in md.lower()
    assert "92" in md  # faithfulness %
