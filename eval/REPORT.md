# GenAcademy RAG — Evaluation Report

## Scores

| Metric | Value |
|---|---|
| Retrieval questions | 12 |
| recall@k | 0.79 |
| precision@k | 0.25 |
| MRR | 0.58 |
| refusal correctness | 1.00 |
| faithfulness % (LLM-judge) | 100% |

## Final rerank/Nebius run

Regenerated on 2026-06-11 with `GENACADEMY_PROVIDER=nebius`,
`NEBIUS_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507`, `GENACADEMY_RERANK_ENABLED=true`,
`GENACADEMY_RERANK_POOL=20`, and `GENACADEMY_RERANK_DEVICE=cpu`.

The generation model was selected from the live Nebius model catalog. `Qwen/Qwen3-30B-A3B-Instruct-2507`
was the fastest candidate that passed the load-bearing JSON grader gate: 5 measured answer-shaped
calls averaged 1491.2 ms, and 10/10 grader-shaped JSON-mode calls parsed cleanly.

## Per-question

| id | category | recall | precision | mrr | refused | faithful |
|---|---|---|---|---|---|---|
| q1 | answerable | 1.00 | 0.20 | 1.00 | False | True |
| q2 | answerable | 1.00 | 0.40 | 1.00 | False | True |
| q3 | answerable | 1.00 | 0.40 | 1.00 | False | True |
| q4 | answerable | 1.00 | 0.20 | 0.50 | False | True |
| q5 | exact_match | 0.00 | 0.00 | 0.00 | False | True |
| q6 | exact_match | 1.00 | 0.40 | 1.00 | False | True |
| q7 | chunking_stress | 1.00 | 0.20 | 0.33 | False | True |
| q8 | chunking_stress | 1.00 | 0.20 | 0.33 | False | True |
| q9 | multi_document | 0.50 | 0.20 | 0.33 | False | True |
| q10 | multi_document | 0.50 | 0.20 | 0.25 | False | True |
| q11 | ambiguous | 1.00 | 0.40 | 1.00 | False | True |
| q12 | ambiguous | 0.50 | 0.20 | 0.25 | False | True |
| q13 | unanswerable | — | — | — | True | — |
| q14 | unanswerable | — | — | — | True | — |
| q15 | unanswerable | — | — | — | True | — |

## Failure analysis

| Symptom | Cause | Fix | Question |
|---|---|---|---|
| Exact-match gold span was not retrieved in top-5, though the answer path no longer refused | ChunkingBoundary - line 53 is a compact Week-2 table row; the fixed-size chunks rank broader overview and adjacent catalog chunks higher than the one-line tagline chunk | Section-aware/table-aware chunking for catalog tables, or adjacent-row stitching for compact table rows | q5 |
| Only one of two required multi-document spans was retrieved | TopKTooSmall - rerank lifted the catalog structured-output row, but the companion `langchain_prompts.py` extraction prompt remained outside top-5 behind notebook/README chunks | For multi-document questions, expand `top_k` or add query decomposition before rerank; keep the reranker as the first improvement because it recovered the catalog side | q9 |
| Only the catalog prompt-resource span was retrieved; the companion README requirements span was missed | TopKTooSmall + cross-repo coverage - top-5 includes the Week-1 catalog prompt rows but not README lines 57-59 for Python/tooling requirements | Multi-hop retrieval or per-repo diversification before final top-5 truncation | q10 |
| Ambiguous "tool use" query retrieved the Week-3 tool-use row but missed the Week-1 related row | Ambiguity - the query can mean curriculum location or prerequisite resource; rerank chose the direct Week-3 "tool use" curriculum hit | Preserve both interpretations with query expansion or diversified final ranking for ambiguous curriculum terms | q12 |

## Model-swap demo

Same question answered by two providers (2026-06-08). Retrieval + embeddings unchanged
(local STEmbedder); only the generation model swaps.

**Question:** Which resource in the Gen Academy catalog covers chunking strategies for RAG?

**openrouter / meta-llama/llama-3.1-70b-instruct**
```
[Chunking Strategies for RAG](https://weaviate.io/blog/chunking-strategies-for-rag), Weaviate
```

**nebius / meta-llama/Llama-3.3-70B-Instruct**
```
The resource that covers chunking strategies for RAG is:
[Chunking Strategies for RAG](https://weaviate.io/blog/chunking-strategies-for-rag) by Weaviate.
```

Both refused=False, same citation, different verbosity. Swap confirmed end-to-end.
Run `GENACADEMY_PROVIDER=openrouter|nebius|openai` to reproduce.
