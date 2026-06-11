# Phase 2 Rerank Eval Delta

**Date:** 2026-06-11
**Corpus:** immutable `eval` Chroma collection
**Gold set:** `src/genacademy_rag/eval/gold/gold_set.yaml`
**Top K:** 5
**Candidate K:** 20
**Rerank model:** `cross-encoder/ms-marco-MiniLM-L6-v2`
**Rerank device for committed run:** `cpu`
**Final generation model for full eval:** `Qwen/Qwen3-30B-A3B-Instruct-2507` on the Nebius preset
**Rerank pool:** `0` = full fused union; `20` = final shipped cap
**Rerank local files only:** true

## Small-N Caveat

This eval has n=12 retrieval-scored questions. One question changes aggregate recall or MRR by about
0.08, so per-question movement is the meaningful evidence. This report does not claim statistical
significance.

## Aggregate Metrics

| Run | recall@k | precision@k | MRR | n |
| --- | ---: | ---: | ---: | ---: |
| Baseline hybrid | 0.67 | 0.22 | 0.55 | 12 |
| Hybrid + rerank, pool=0 | 0.79 | 0.25 | 0.58 | 12 |
| Hybrid + rerank, pool=20 | 0.79 | 0.25 | 0.58 | 12 |

## Retrieval Latency

| Run | mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| Baseline hybrid | 121 | 46 | 1227 |
| Hybrid + rerank, pool=0 | 680 | 534 | 1469 |
| Hybrid + rerank, pool=20 | 429 | 334 | 1165 |

The baseline latency was re-measured on 2026-06-11 and its p95 now exceeds the pool=20 rerank p95;
with only 15 questions, the p95 column is noisy and this anomaly should not be read as a durable
latency ordering.

## Per-Question Movement

| Question | Category | Baseline recall | Pool=0 recall | Pool=20 recall | Baseline MRR | Pool=0 MRR | Pool=20 MRR | Movement at pool=20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q1 | answerable | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | unchanged |
| q2 | answerable | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | unchanged |
| q3 | answerable | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | unchanged |
| q4 | answerable | 1.00 | 1.00 | 1.00 | 0.33 | 0.50 | 0.50 | helped |
| q5 | exact_match | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | unchanged |
| q6 | exact_match | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | unchanged |
| q7 | chunking_stress | 0.00 | 1.00 | 1.00 | 0.00 | 0.25 | 0.33 | helped |
| q8 | chunking_stress | 1.00 | 1.00 | 1.00 | 0.50 | 0.33 | 0.33 | hurt |
| q9 | multi_document | 0.50 | 0.50 | 0.50 | 0.50 | 0.33 | 0.33 | hurt |
| q10 | multi_document | 0.50 | 0.50 | 0.50 | 0.25 | 0.25 | 0.25 | unchanged |
| q11 | ambiguous | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | unchanged |
| q12 | ambiguous | 0.00 | 0.50 | 0.50 | 0.00 | 0.25 | 0.25 | helped |
| q13 | unanswerable | n/a | n/a | n/a | n/a | n/a | n/a | unchanged |
| q14 | unanswerable | n/a | n/a | n/a | n/a | n/a | n/a | unchanged |
| q15 | unanswerable | n/a | n/a | n/a | n/a | n/a | n/a | unchanged |

## Refusal-Behavior Proxy From Top-K Membership

Provenance: this proxy table is carried forward from the 2026-06-09 pool=0 rerank run; it was not
regenerated for the final pool=20 measurement.

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

Rerank improved retrieval quality on this run: recall rose from 0.67 to 0.79, precision from 0.22 to
0.25, and MRR from 0.55 to 0.58. The final `GENACADEMY_RERANK_POOL=20` cap preserved the full-union
recall win while lowering mean retrieval latency from 680.258 ms to 429.110 ms versus pool=0 on the
CPU-pinned run. The latency columns are from one 15-question command per configuration and include
local warm-up noise, so the reliable conclusion is directional: pool=20 keeps the retrieval win and is
the better serving default for the Space.

The full 15-question eval was regenerated on the Nebius preset with
`NEBIUS_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507`, `GENACADEMY_RERANK_ENABLED=true`, and
`GENACADEMY_RERANK_POOL=20`. It produced recall@k 0.79, precision@k 0.25, MRR 0.58, refusal
correctness 1.00, and LLM-judge faithfulness 100% in `eval/REPORT.md`.

Caveat: `scripts/run_eval.py` uses the same provider/model for generation and the LLM judge, so this
run was judged by `Qwen/Qwen3-30B-A3B-Instruct-2507`; the 58%->100% faithfulness and 0.73->1.00
refusal deltas versus the prior `meta-llama/Llama-3.3-70B-Instruct`-judged run are not attributable
to rerank alone.
