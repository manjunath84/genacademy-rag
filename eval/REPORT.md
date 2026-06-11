# GenAcademy RAG — Evaluation Report

## Scores

| Metric | Value |
|---|---|
| Retrieval questions | 12 |
| recall@k | 0.67 |
| precision@k | 0.22 |
| MRR | 0.55 |
| refusal correctness | 0.73 |
| faithfulness % (LLM-judge) | 58% |

## Answer UX prompt rerun

Regenerated on 2026-06-10 with `GENACADEMY_PROVIDER=nebius` and judge model
`meta-llama/Llama-3.3-70B-Instruct` after changing the answer prompt from terse
single-sentence answers to overview paragraph + key-point bullets.

| Run | recall@k | precision@k | MRR | faithfulness |
|---|---|---|---|---|
| Before answer-card prompt change | 0.67 | 0.22 | 0.55 | 58% |
| After answer-card prompt change | 0.67 | 0.22 | 0.55 | 58% |

Faithfulness delta: **0 percentage points**. Retrieval metrics are unchanged, as required; this
slice did not change retrieval, chunking, grader logic, or the gold set.

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
| q8 | chunking_stress | 1.00 | 0.20 | 0.50 | False | False |
| q9 | multi_document | 0.50 | 0.20 | 0.50 | False | True |
| q10 | multi_document | 0.50 | 0.20 | 0.25 | True | False |
| q11 | ambiguous | 1.00 | 0.40 | 1.00 | True | False |
| q12 | ambiguous | 0.00 | 0.00 | 0.00 | False | True |
| q13 | unanswerable | — | — | — | True | — |
| q14 | unanswerable | — | — | — | True | — |
| q15 | unanswerable | — | — | — | True | — |

## Failure analysis

| Symptom | Cause | Fix | Question |
|---|---|---|---|
| Exact-match question refused; gold span at a compact table row not retrieved | ChunkingBoundary — line 53 ("RAG & Context Engineering" tagline) is a one-cell markdown table entry; 1000-char window strips the surrounding section header into the adjacent chunk, leaving this chunk with cosine < 0.20 | Increase chunk_overlap 150→300 chars so section headers bleed into the next chunk; or lower cosine_threshold to 0.15 | q5 |
| Comprehensive-list question refused; gold span covers 13 lines (lines 29-41, ~700 chars) | ChunkingBoundary — fixed 1000-char window splits the Week-1 prereq table across two chunks; neither half reaches threshold on its own | Switch to sentence/table-boundary chunker for catalog sections; Phase-2 multi-chunk stitching via reranker | q7 |
| Retrieval succeeded (recall=1.00) but judge flags context as truncated | ChunkingBoundary — Week-6 resources table exceeds 1000 chars; retrieved chunk holds only the table header + first row; judge scores incomplete context as unfaithful (score=1) | Raise chunk_size to 1500 for table-heavy sections; Phase-2 stitches top-2 adjacent chunks before generation | q8 |
| Only 1 of 2 required gold spans retrieved; code-file span missed | TopKTooSmall — top_k=5 spread across two repos; langchain_prompts.py lines 13-43 ranked 6th against denser catalog text | Raise top_k to 8 for multi-document categories; Phase-2 cross-encoder reranker consolidates candidates across repos | q9 |
| Partial retrieval (recall=0.50) triggered incorrect refusal | TopKTooSmall + RefusalFalsePositive — second gold span (2-line README entry) ranked outside top-5; low coverage depressed answer confidence below refusal threshold despite finding the first span | Raise top_k to 8; gate refusal on max chunk score (≥ 0.35) rather than mean confidence so partial-but-useful retrieval still answers | q10 |
| All 3 gold spans retrieved (recall=1.00) but system refused to answer | RefusalFalsePositive — broad query "what does it say about embeddings?" disperses cosine scores across many chunks; mean confidence falls below refusal threshold even though individual chunks score well | Gate refusal on max chunk score, not mean; add query-intent classifier to suppress refusal on broad informational questions | q11 |

## Model-swap demo

Same question answered by two providers (2026-06-08). Retrieval + embeddings unchanged (local STEmbedder); only the generation model swaps.

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
