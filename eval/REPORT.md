# GenAcademy RAG — Evaluation Report

## Scores

| Metric | Value |
|---|---|
| Retrieval questions | 12 |
| recall@k | 0.67 |
| precision@k | 0.22 |
| MRR | 0.55 |
| refusal correctness | 0.67 |
| faithfulness % (citation-grounding fallback) | 58% |

## Per-question

| id | category | recall | precision | mrr | refused | faithful |
|---|---|---|---|---|---|---|
| q1 | answerable | 1.00 | 0.20 | 1.00 | False | True |
| q2 | answerable | 1.00 | 0.40 | 1.00 | False | True |
| q3 | answerable | 1.00 | 0.40 | 1.00 | False | True |
| q4 | answerable | 1.00 | 0.20 | 0.33 | False | True |
| q5 | exact_match | 0.00 | 0.00 | 0.00 | True | False |
| q6 | exact_match | 1.00 | 0.40 | 1.00 | False | True |
| q7 | chunking_stress | 0.00 | 0.00 | 0.00 | True | False |
| q8 | chunking_stress | 1.00 | 0.20 | 0.50 | False | True |
| q9 | multi_document | 0.50 | 0.20 | 0.50 | False | True |
| q10 | multi_document | 0.50 | 0.20 | 0.25 | True | False |
| q11 | ambiguous | 1.00 | 0.40 | 1.00 | True | False |
| q12 | ambiguous | 0.00 | 0.00 | 0.00 | True | False |
| q13 | unanswerable | — | — | — | True | — |
| q14 | unanswerable | — | — | — | True | — |
| q15 | unanswerable | — | — | — | True | — |

## Failure analysis

| Symptom | Cause | Fix | Question |
|---|---|---|---|
