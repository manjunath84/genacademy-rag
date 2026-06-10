# Phase 2 Section-Aware Chunking Delta

**Date:** 2026-06-10
**Primary comparison:** fixed-size chunking vs section-aware chunking
**Rerank:** disabled for both primary runs
**Gold set:** `src/genacademy_rag/eval/gold/gold_set.yaml`
**Baseline collection:** `eval` (chunker=fixed, chunk_size=1000, chunk_overlap=150)
**Candidate collection:** `eval_section` (chunker=section, section_chunk_max_chars=1500, section_chunk_overlap=150)
**Shared settings:** top_k=5, candidate_k=20, embed model all-MiniLM-L6-v2

## Summary

| Run | recall@k | precision@k | MRR | mean retrieval ms | p50 retrieval ms | p95 retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed baseline | 0.67 | 0.22 | 0.55 | 301.555 | 88.417 | 1387.725 |
| section-aware | 0.64 | 0.18 | 0.34 | 59.975 | 52.732 | 217.876 |

## Corpus Shape

| Run | collection | chunker | chunk count |
| --- | --- | --- | ---: |
| fixed baseline | eval | fixed | 53 |
| section-aware | eval_section | section | 73 |

Section-aware chunking produced more, smaller heading-bounded chunks after enforcing heading
boundaries. Latency was lower in this run, but with 12 scored queries and separate Python processes
for each eval command, the latency columns are warm-up/noise dominated. They should not be read as a
quality win. Recall, precision, and MRR all dropped.

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

## Stress and Multi-Document Questions

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
the recall losses on q9 and q11 and rank regressions on q2, q4, q6, and q8.

One confound to keep in mind when reading the table: this A/B varies chunk *size* along with chunk
*strategy*. Section chunks may reach `max_chars=1500` (~375 tokens at ~4 chars/token), past the
embedder's 256-token window, so the tail of any near-max section chunk is invisible to dense
retrieval even though it is stored in the chunk text and citation span. Fixed chunks at 1000 chars
stay under the cap. Some of the MRR drop may be embedder tail-truncation rather than the heading
boundaries themselves; a follow-up run with `section_chunk_max_chars=1000` would separate the two.

Mean retrieval latency fell from 301.555 ms to 59.975 ms and p95 fell from 1387.725 ms to
217.876 ms in this run, while p50 fell from 88.417 ms to 52.732 ms. With 12 queries and one process
per eval command, these columns are dominated by model/query warm-up noise; the honest latency claim
is "not the decision driver," not "improved."

This is a 12-question retrieval-scored eval, so the results are directional rather than statistically
strong. The honest conclusion is that strict heading-bounded chunking is easy to reason about but too
fragmented for the current retriever settings.

## Recommendation

Keep section-aware chunking implemented but disabled by default.

The next experiment should focus on preserving local rank quality while recovering section context.
Two plausible follow-ups are:

- report a secondary section-aware plus rerank run, because rerank may offset the MRR drop from larger
  or more fragmented chunks
- re-run with `section_chunk_max_chars=1000` to remove the embedder tail-truncation confound and
  isolate the effect of heading boundaries alone
- tune the section chunker to carry a compact heading prelude or adjacent heading context without
  turning every heading boundary into an isolated retrieval island
