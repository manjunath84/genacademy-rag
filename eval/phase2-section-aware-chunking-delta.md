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
| section-aware | 0.64 | 0.18 | 0.34 | 80.084 | 80.474 | 228.895 |

## Corpus Shape

| Run | collection | chunker | chunk count |
| --- | --- | --- | ---: |
| fixed baseline | eval | fixed | 53 |
| section-aware | eval_section | section | 73 |

Section-aware chunking produced more, smaller heading-bounded chunks after enforcing heading
boundaries. That reduced latency in this run, but it also lowered recall, precision, and MRR.

## Per-Question Movement

| ID | Category | Baseline recall | Section recall | Baseline MRR | Section MRR | Movement |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| q1 | answerable | 1.00 | 1.00 | 1.00 | 1.00 | unchanged |
| q2 | answerable | 1.00 | 1.00 | 1.00 | 0.33 | rank regressed |
| q3 | answerable | 1.00 | 1.00 | 1.00 | 1.00 | unchanged |
| q4 | answerable | 1.00 | 1.00 | 0.33 | 0.25 | rank regressed |
| q5 | exact_match | 0.00 | 0.00 | 0.00 | 0.00 | unchanged miss |
| q6 | exact_match | 1.00 | 1.00 | 1.00 | 0.20 | rank regressed |
| q7 | chunking_stress | 0.00 | 0.00 | 0.00 | 0.00 | unchanged miss |
| q8 | chunking_stress | 1.00 | 1.00 | 0.50 | 0.33 | rank regressed |
| q9 | multi_document | 0.50 | 0.00 | 0.50 | 0.00 | recall and rank regressed |
| q10 | multi_document | 0.50 | 0.50 | 0.25 | 0.25 | unchanged |
| q11 | ambiguous | 1.00 | 0.67 | 1.00 | 0.50 | recall and rank regressed |
| q12 | ambiguous | 0.00 | 0.50 | 0.00 | 0.20 | recall and rank improved |

## Chunking-Stress Questions

| ID | Expected pressure | Movement |
| --- | --- | --- |
| q5 | compact markdown table row with section context | unchanged miss |
| q7 | prerequisite table split across fixed windows | unchanged miss |
| q8 | Week 6 resource table truncation | recall unchanged, rank worsened from 0.50 to 0.33 MRR |
| q9 | multi-document top-k pressure | worsened from 0.50 to 0.00 recall and 0.50 to 0.00 MRR |
| q10 | multi-document top-k pressure | unchanged at 0.50 recall and 0.25 MRR |

## Interpretation

Section-aware chunking is not a win in this configuration. It lowers aggregate recall from 0.67 to
0.64, precision from 0.22 to 0.18, and MRR from 0.55 to 0.34. It does not fix the q5 or q7
chunking-boundary misses that motivated the slice. It improves q12, but that gain is outweighed by
the q9 recall loss and rank regressions on q2, q4, q6, q8, and q11.

The latency signal improves, likely because the section-aware collection's headings and smaller chunks
produce a cheaper BM25/dense candidate set in this run. Mean retrieval latency fell from 90.918 ms to
80.084 ms and p95 fell from 360.648 ms to 228.895 ms, while p50 was effectively unchanged.

This is a 12-question retrieval-scored eval, so the results are directional rather than statistically
strong. The honest conclusion is that strict heading-bounded chunking is easy to reason about but too
fragmented for the current retriever settings.

## Recommendation

Keep section-aware chunking implemented but disabled by default.

The next experiment should focus on preserving local rank quality while recovering section context.
Two plausible follow-ups are:

- report a secondary section-aware plus rerank run, because rerank may offset the MRR drop from larger
  or more fragmented chunks
- tune the section chunker to carry a compact heading prelude or adjacent heading context without
  turning every heading boundary into an isolated retrieval island
