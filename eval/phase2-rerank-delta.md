# Phase 2 Rerank Eval Delta

**Date:** 2026-06-09
**Corpus:** immutable `eval` Chroma collection
**Gold set:** `src/genacademy_rag/eval/gold/gold_set.yaml`
**Top K:** 5
**Candidate K:** 20
**Rerank model:** `cross-encoder/ms-marco-MiniLM-L6-v2`
**Rerank device for committed run:** `cpu`
**Rerank pool:** 0 = full fused union when 0
**Rerank local files only:** true

## Small-N Caveat

This eval has n=12 retrieval-scored questions. One question changes aggregate recall or MRR by about 0.08, so per-question movement is the meaningful evidence. This report does not claim statistical significance.

## Aggregate Metrics

| Run | recall@k | precision@k | MRR | n |
| --- | ---: | ---: | ---: | ---: |
| Baseline hybrid | 0.67 | 0.22 | 0.55 | 12 |
| Hybrid + rerank | 0.79 | 0.25 | 0.58 | 12 |

## Retrieval Latency

| Run | mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| Baseline hybrid | 82.864 | 77.616 | 286.264 |
| Hybrid + rerank | 565.625 | 509.416 | 886.352 |

## Per-Question Movement

| Question | Category | Baseline recall | Rerank recall | Baseline MRR | Rerank MRR | Baseline top chunk | Rerank top chunk | Movement |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| q1 | answerable | 1.00 | 1.00 | 1.00 | 1.00 | awesome-agentic-ai-resources/README.md@5dfb869::0 | awesome-agentic-ai-resources/README.md@5dfb869::0 | unchanged |
| q2 | answerable | 1.00 | 1.00 | 1.00 | 1.00 | Mastering-Agentic-AI-Week1/Langchain Basics/README.md@3aa31df::0 | Mastering-Agentic-AI-Week1/Langchain Basics/README.md@3aa31df::1 | unchanged |
| q3 | answerable | 1.00 | 1.00 | 1.00 | 1.00 | awesome-agentic-ai-resources/README.md@5dfb869::10 | awesome-agentic-ai-resources/README.md@5dfb869::10 | unchanged |
| q4 | answerable | 1.00 | 1.00 | 0.33 | 0.50 | Mastering-Agentic-AI-Week1/Langchain Basics/Langchain_Fundamentals.ipynb@3aa31df::2 | Mastering-Agentic-AI-Week1/Langchain Basics/Langchain_Fundamentals.ipynb@3aa31df::2 | helped |
| q5 | exact_match | 0.00 | 0.00 | 0.00 | 0.00 | awesome-agentic-ai-resources/README.md@5dfb869::0 | awesome-agentic-ai-resources/README.md@5dfb869::0 | unchanged |
| q6 | exact_match | 1.00 | 1.00 | 1.00 | 1.00 | awesome-agentic-ai-resources/README.md@5dfb869::24 | awesome-agentic-ai-resources/README.md@5dfb869::24 | unchanged |
| q7 | chunking_stress | 0.00 | 1.00 | 0.00 | 0.25 | awesome-agentic-ai-resources/README.md@5dfb869::0 | awesome-agentic-ai-resources/README.md@5dfb869::0 | helped |
| q8 | chunking_stress | 1.00 | 1.00 | 0.50 | 0.33 | awesome-agentic-ai-resources/README.md@5dfb869::0 | awesome-agentic-ai-resources/README.md@5dfb869::1 | hurt |
| q9 | multi_document | 0.50 | 0.50 | 0.50 | 0.33 | Mastering-Agentic-AI-Week1/Langchain Basics/Langchain_Fundamentals.ipynb@3aa31df::0 | Mastering-Agentic-AI-Week1/Langchain Basics/README.md@3aa31df::0 | hurt |
| q10 | multi_document | 0.50 | 0.50 | 0.25 | 0.25 | Mastering-Agentic-AI-Week1/Langchain Basics/README.md@3aa31df::0 | Mastering-Agentic-AI-Week1/Langchain Basics/README.md@3aa31df::0 | unchanged |
| q11 | ambiguous | 1.00 | 1.00 | 1.00 | 1.00 | awesome-agentic-ai-resources/README.md@5dfb869::8 | awesome-agentic-ai-resources/README.md@5dfb869::0 | unchanged |
| q12 | ambiguous | 0.00 | 0.50 | 0.00 | 0.25 | awesome-agentic-ai-resources/README.md@5dfb869::1 | awesome-agentic-ai-resources/README.md@5dfb869::1 | helped |
| q13 | unanswerable | n/a | n/a | n/a | n/a | awesome-agentic-ai-resources/README.md@5dfb869::12 | awesome-agentic-ai-resources/README.md@5dfb869::0 | unchanged |
| q14 | unanswerable | n/a | n/a | n/a | n/a | awesome-agentic-ai-resources/README.md@5dfb869::24 | awesome-agentic-ai-resources/README.md@5dfb869::24 | unchanged |
| q15 | unanswerable | n/a | n/a | n/a | n/a | awesome-agentic-ai-resources/README.md@5dfb869::0 | awesome-agentic-ai-resources/README.md@5dfb869::0 | unchanged |

## Refusal-Behavior Proxy From Top-K Membership

| Question | Baseline max cosine | Rerank max cosine | Baseline fallback answerable @0.2 | Rerank fallback answerable @0.2 | Change |
| --- | ---: | ---: | --- | --- | --- |
| q1 | 0.43 | 0.43 | True | True | unchanged |
| q2 | 0.47 | 0.47 | True | True | unchanged |
| q3 | 0.54 | 0.54 | True | True | unchanged |
| q4 | 0.62 | 0.62 | True | True | unchanged |
| q5 | 0.33 | 0.33 | True | True | unchanged |
| q6 | 0.20 | 0.20 | True | True | unchanged |
| q7 | 0.40 | 0.40 | True | True | unchanged |
| q8 | 0.54 | 0.54 | True | True | unchanged |
| q9 | 0.43 | 0.43 | True | True | unchanged |
| q10 | 0.48 | 0.48 | True | True | unchanged |
| q11 | 0.36 | 0.38 | True | True | unchanged |
| q12 | 0.39 | 0.39 | True | True | unchanged |
| q13 | 0.30 | 0.28 | True | True | unchanged |
| q14 | 0.42 | 0.42 | True | True | unchanged |
| q15 | 0.51 | 0.51 | True | True | unchanged |

## Interpretation

Rerank improved retrieval quality on this run: recall, precision, and MRR all increased. The cost is material: p95 retrieval latency rose from 286.264 ms to 886.352 ms on the CPU-pinned run. The likely cause is that the committed eval scores full real chunks of about 1000 characters each on CPU fp32, while the Phase A synthetic timing used much shorter query-passage pairs. This exceeds the Phase 2 300 ms caution threshold, so keep rerank disabled by default unless demo quality is worth the local latency. Fallback-proxy answerability did not change at threshold 0.2 in this run.
