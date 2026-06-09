"""Faithfulness eval (depth add-on, cuttable). llm_judge_score = pinned-prompt LLM-as-judge at
temp 0 (raw outputs saved by the report). citation_grounding_score = the zero-LLM fallback that
always ships: do the answer's content words actually appear in the retrieved chunks?"""
from __future__ import annotations

import json
import re

from genacademy_rag.core.types import RetrievedChunk

FAITHFULNESS_JUDGE_SYSTEM = "You are a strict faithfulness judge. Reply ONLY with a JSON object."
FAITHFULNESS_JUDGE_USER = (
    "Question:\n{question}\n\nAnswer to judge:\n{answer}\n\nRetrieved context (ground truth):\n"
    "{context}\n\nIs every claim in the answer supported by the context? Return exactly "
    '{{"faithful": <true|false>, "hallucinated_claims": [<strings>], "score": <1-5 integer>}}.'
)

_STOP = {"the", "a", "an", "of", "to", "and", "is", "are", "in", "on", "for", "it", "that",
         "this", "with", "as", "be", "by", "or", "what", "which", "does", "do"}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def citation_grounding_score(answer: str, retrieved: list[RetrievedChunk],
                              min_overlap: float = 0.6) -> bool:
    ans = _content_words(answer)
    if not ans:
        return True
    ctx: set[str] = set()
    for r in retrieved:
        ctx |= _content_words(r.chunk.text)
    return len(ans & ctx) / len(ans) >= min_overlap


def llm_judge_score(question: str, answer: str, retrieved: list[RetrievedChunk], provider) -> dict:
    context = "\n---\n".join(r.chunk.text for r in retrieved)
    raw = provider.generate(
        [{"role": "system", "content": FAITHFULNESS_JUDGE_SYSTEM},
         {"role": "user", "content": FAITHFULNESS_JUDGE_USER.format(
             question=question, answer=answer, context=context)}],
        json_mode=True, max_tokens=256, temperature=0.0)
    parsed = json.loads(raw)
    return {"faithful": bool(parsed["faithful"]),
            "hallucinated_claims": parsed.get("hallucinated_claims", []),
            "score": int(parsed.get("score", 0)), "raw": raw}
