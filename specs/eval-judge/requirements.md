# Faithfulness Judge Prompt (verbatim)

Pinned here in sync with `src/genacademy_rag/eval/faithfulness_eval.py`.

## System prompt

```
You are a strict faithfulness judge. Reply ONLY with a JSON object.
```

## User prompt template

```
Question:
{question}

Answer to judge:
{answer}

Retrieved context (ground truth):
{context}

Is every claim in the answer supported by the context? Return exactly
{"faithful": <true|false>, "hallucinated_claims": [<strings>], "score": <1-5 integer>}.
```

## Expected JSON response shape

```json
{
  "faithful": true,
  "hallucinated_claims": [],
  "score": 5
}
```

- `faithful`: boolean — true iff every claim is grounded in the retrieved context
- `hallucinated_claims`: list of strings — specific unsupported claims (empty when faithful=true)
- `score`: integer 1–5 — overall faithfulness score

## Parameters

- `json_mode=True`, `temperature=0.0`, `max_tokens=256`
- Runs as the LLM-judge path; `citation_grounding_score()` is the zero-LLM fallback.
