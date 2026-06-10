# Phase 2 Section-Aware Chunking Delta

**Date:** 2026-06-10
**Primary comparison:** fixed-size chunking vs section-aware chunking
**Rerank:** disabled for both primary runs
**Gold set:** `src/genacademy_rag/eval/gold/gold_set.yaml`
**Baseline collection:** `eval`
**Candidate collection:** `eval_section`

## Summary

| Run | recall@k | precision@k | MRR | mean retrieval ms | p50 retrieval ms | p95 retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed baseline | 0.67 | 0.22 | 0.55 | 90.918 | 83.546 | 360.648 |
| section-aware | 0.68 | 0.22 | 0.47 | 100.141 | 81.082 | 516.260 |

## Corpus Shape

| Run | collection | chunker | chunk count |
| --- | --- | --- | ---: |
| fixed baseline | eval | fixed | 53 |
| section-aware | eval_section | section | 38 |

Section-aware chunking produced fewer, larger chunks. That slightly improved aggregate recall, but it
also lowered rank quality enough to reduce MRR.

## Per-Question Movement

| ID | Category | Baseline recall | Section recall | Baseline MRR | Section MRR | Movement |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| q1 | answerable | 1.00 | 1.00 | 1.00 | 1.00 | unchanged |
| q2 | answerable | 1.00 | 1.00 | 1.00 | 1.00 | unchanged |
| q3 | answerable | 1.00 | 1.00 | 1.00 | 1.00 | unchanged |
| q4 | answerable | 1.00 | 1.00 | 0.33 | 0.50 | rank improved |
| q5 | exact_match | 0.00 | 0.00 | 0.00 | 0.00 | unchanged miss |
| q6 | exact_match | 1.00 | 1.00 | 1.00 | 0.50 | rank regressed |
| q7 | chunking_stress | 0.00 | 0.00 | 0.00 | 0.00 | unchanged miss |
| q8 | chunking_stress | 1.00 | 1.00 | 0.50 | 0.33 | rank regressed |
| q9 | multi_document | 0.50 | 0.50 | 0.50 | 0.25 | rank regressed |
| q10 | multi_document | 0.50 | 1.00 | 0.25 | 0.50 | recall and rank improved |
| q11 | ambiguous | 1.00 | 0.67 | 1.00 | 0.50 | recall and rank regressed |
| q12 | ambiguous | 0.00 | 0.00 | 0.00 | 0.00 | unchanged miss |

## Chunking-Stress Questions

| ID | Expected pressure | Movement |
| --- | --- | --- |
| q5 | compact markdown table row with section context | unchanged miss |
| q7 | prerequisite table split across fixed windows | unchanged miss |
| q8 | Week 6 resource table truncation | recall unchanged, rank worsened from 0.50 to 0.33 MRR |
| q9 | multi-document top-k pressure | recall unchanged, rank worsened from 0.50 to 0.25 MRR |
| q10 | multi-document top-k pressure | improved from 0.50 to 1.00 recall and 0.25 to 0.50 MRR |

## Interpretation

Section-aware chunking is not a clear win in this configuration. It improves aggregate recall from
0.67 to 0.68 and fixes one multi-document recall miss on q10, but it does not fix the q5 or q7
chunking-boundary misses that motivated the slice. It also lowers MRR from 0.55 to 0.47, with visible
rank regressions on q6, q8, q9, and q11.

The latency signal is also mixed. Mean retrieval latency rose from 90.918 ms to 100.141 ms and p95 rose
from 360.648 ms to 516.260 ms, while p50 was effectively unchanged.

This is a 12-question retrieval-scored eval, so the results are directional rather than statistically
strong. The honest conclusion is that the current sectioning heuristic changes corpus shape and can
recover some context, but it needs additional ranking or chunk-boundary tuning before becoming the
default.

## Recommendation

Keep section-aware chunking implemented but disabled by default.

The next experiment should focus on preserving local rank quality while recovering section context.
Two plausible follow-ups are:

- report a secondary section-aware plus rerank run, because rerank may offset the MRR drop from larger
  chunks
- tune the section chunker to split large sections more tightly around table/list blocks instead of
  grouping broad section spans
