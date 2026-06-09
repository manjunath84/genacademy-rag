from genacademy_rag.core.analytics import usage_summary


def test_usage_summary_empty_rows():
    summary = usage_summary([])
    assert summary["total_queries"] == 0
    assert summary["refusal_rate"] == 0.0
    assert summary["fallback_rate"] == 0.0
    assert summary["latency_p50_ms"] == 0
    assert summary["latency_p95_ms"] == 0
    assert summary["top_questions"] == []
    assert summary["queries_by_day"] == []


def test_usage_summary_rates_percentiles_top_questions_and_days():
    rows = [
        {
            "ts": "2026-06-09 10:00:00",
            "question": "What is RAG?",
            "refused": 0,
            "used_fallback": 0,
            "latency_ms": 100,
        },
        {
            "ts": "2026-06-09 10:01:00",
            "question": "What is RAG?",
            "refused": 1,
            "used_fallback": 1,
            "latency_ms": 200,
        },
        {
            "ts": "2026-06-10 10:00:00",
            "question": "What is BM25?",
            "refused": 0,
            "used_fallback": 0,
            "latency_ms": 300,
        },
        {
            "ts": "2026-06-10 10:01:00",
            "question": "What is CSRF?",
            "refused": 0,
            "used_fallback": 1,
            "latency_ms": 400,
        },
    ]
    summary = usage_summary(rows, top_n=2)
    assert summary["total_queries"] == 4
    assert summary["refusal_rate"] == 0.25
    assert summary["fallback_rate"] == 0.5
    assert summary["latency_p50_ms"] == 250
    assert summary["latency_p95_ms"] == 385
    assert summary["top_questions"] == [
        {"question": "What is RAG?", "count": 2},
        {"question": "What is BM25?", "count": 1},
    ]
    assert summary["queries_by_day"] == [
        {"day": "2026-06-09", "count": 2},
        {"day": "2026-06-10", "count": 2},
    ]
