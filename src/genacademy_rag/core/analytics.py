"""Pure usage analytics for the admin dashboard."""
from __future__ import annotations

from collections import Counter
from math import ceil, floor


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lower = floor(pos)
    upper = ceil(pos)
    if lower == upper:
        return int(ordered[lower])
    weight = pos - lower
    return int(round(ordered[lower] * (1 - weight) + ordered[upper] * weight))


def usage_summary(rows: list[dict], *, top_n: int = 5) -> dict:
    total = len(rows)
    if total == 0:
        return {
            "total_queries": 0,
            "refusal_rate": 0.0,
            "fallback_rate": 0.0,
            "latency_p50_ms": 0,
            "latency_p95_ms": 0,
            "top_questions": [],
            "queries_by_day": [],
        }
    refused = sum(1 for row in rows if int(row.get("refused") or 0))
    fallback = sum(1 for row in rows if int(row.get("used_fallback") or 0))
    latencies = [int(row.get("latency_ms") or 0) for row in rows]
    questions = Counter(str(row.get("question") or "") for row in rows)
    days = Counter(str(row.get("ts") or "")[:10] for row in rows)
    return {
        "total_queries": total,
        "refusal_rate": refused / total,
        "fallback_rate": fallback / total,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "top_questions": [
            {"question": question, "count": count}
            for question, count in questions.most_common(top_n)
        ],
        "queries_by_day": [
            {"day": day, "count": days[day]}
            for day in sorted(days)
            if day
        ],
    }
