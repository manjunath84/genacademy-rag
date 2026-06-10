from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk
from genacademy_rag.eval.gold_schema import GoldQuestion, GoldSpan
from genacademy_rag.eval.retrieval_eval import (
    aggregate,
    build_retrieval_eval_payload,
    chunk_matches_span,
    latency_summary,
    score_question,
)


def _rc(line_start, line_end, commit="abc123", text="x"):
    cit = Citation(doc_id="d1", title="README.md", source_type="github",
                   repo="awesome-agentic-ai-resources", file_path="README.md",
                   commit_hash=commit, line_start=line_start, line_end=line_end)
    return RetrievedChunk(chunk=Chunk(chunk_id=f"d1::{line_start}", doc_id="d1",
                                      ordinal=line_start, text=text, citation=cit), score=1.0)


def test_chunk_matches_span_requires_overlap_and_commit():
    span = GoldSpan(repo="awesome-agentic-ai-resources", file_path="README.md",
                    commit_hash="abc123", line_start=10, line_end=20)
    assert chunk_matches_span(_rc(8, 15).chunk, span)            # overlaps 10-15
    assert not chunk_matches_span(_rc(30, 40).chunk, span)       # no overlap
    # commit mismatch -> no leak
    assert not chunk_matches_span(_rc(8, 15, commit="WRONG").chunk, span)


def test_recall_precision_mrr_on_a_hit_at_rank_2():
    q = GoldQuestion(id="q1", question="x", category="answerable", answerable=True,
                     gold=[GoldSpan("awesome-agentic-ai-resources", "README.md", "abc123", 10, 20)])
    retrieved = [_rc(30, 40), _rc(12, 18), _rc(50, 60)]   # gold is at rank 2
    s = score_question(q, retrieved, k=5)
    assert s["recall"] == 1.0
    assert s["precision"] == 1 / 5           # 1 relevant of k=5 slots
    assert s["mrr"] == 1 / 2                 # first relevant at rank 2


def test_unanswerable_question_excluded_from_retrieval_metrics():
    q = GoldQuestion(id="q2", question="x", category="unanswerable", answerable=False, gold=[])
    s = score_question(q, [_rc(1, 2)], k=5)
    assert s["recall"] is None               # not a retrieval-scored question
    agg = aggregate([s])
    assert agg["n_retrieval_questions"] == 0


def test_latency_summary_reports_mean_p50_and_p95():
    summary = latency_summary([10.0, 30.0, 20.0, 40.0])

    assert summary == {
        "retrieval_ms_mean": 25.0,
        "retrieval_ms_p50": 25.0,
        "retrieval_ms_p95": 40.0,
    }


def test_retrieval_eval_payload_includes_metrics_rows_latency_and_config():
    rows = [
        {"id": "q1", "recall": 1.0, "precision": 0.2, "mrr": 1.0, "retrieval_ms": 10.0},
        {"id": "q2", "recall": 0.0, "precision": 0.0, "mrr": 0.0, "retrieval_ms": 30.0},
    ]
    payload = build_retrieval_eval_payload(
        metrics={"recall@k": 0.5, "precision@k": 0.1, "mrr": 0.5, "n_retrieval_questions": 2},
        rows=rows,
        config={"rerank_enabled": True, "rerank_pool": 0, "rerank_device": "cpu"},
    )

    assert payload["metrics"]["recall@k"] == 0.5
    assert payload["questions"] == rows
    assert payload["latency"] == {
        "retrieval_ms_mean": 20.0,
        "retrieval_ms_p50": 20.0,
        "retrieval_ms_p95": 30.0,
    }
    assert payload["config"] == {
        "rerank_enabled": True,
        "rerank_pool": 0,
        "rerank_device": "cpu",
    }
