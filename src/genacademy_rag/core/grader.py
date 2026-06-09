"""Refusal grader. Primary: JSON-mode LLM call (spike confirmed it works on the open model) whose
`answerable` field IS the decision; `confidence` is only a 1-5 bucket reported downstream. Fallback
(on any malformed/unparseable response): max cosine similarity of the retrieved set vs a calibrated
threshold. Load-bearing: if the context doesn't support an answer we refuse, never answer from
priors (AGENTS.md §3). `answerable` is parsed strictly so a stringified `"false"` cannot flip a
refusal into an answer (see json_utils.strict_bool)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from genacademy_rag.core.json_utils import strict_bool
from genacademy_rag.core.types import RetrievedChunk

GRADER_SYSTEM = "You are a strict grader. Reply ONLY with a JSON object."
logger = logging.getLogger(__name__)
GRADER_USER_TMPL = (
    "Question:\n{question}\n\n"
    "Retrieved context (the ONLY allowed source):\n{context}\n\n"
    'Decide if the question can be answered FROM THIS CONTEXT ALONE. '
    'Return exactly {{"answerable": <true|false>, "confidence": <1-5 integer>}}. '
    "answerable=false if the context does not contain the answer."
)


@dataclass(frozen=True)
class Grade:
    answerable: bool
    confidence: int
    used_fallback: bool = False


def cosine_fallback_grade(retrieved: list[RetrievedChunk], threshold: float) -> Grade:
    # r.score is the cosine similarity carried by HybridRetriever (Task 6), in [-1, 1] — NOT the
    # RRF rank score. Calibrate `threshold` (design §7) on 3-5 held-out questions against THIS
    # signal.
    top = max((r.score for r in retrieved), default=0.0)
    answerable = top >= threshold
    # Map the top score into a 1-5 confidence bucket for the report.
    confidence = max(1, min(5, int(round(top * 5)))) if answerable else 1
    return Grade(answerable=answerable, confidence=confidence, used_fallback=True)


def grade_answerability(question: str, retrieved: list[RetrievedChunk], provider, *,
                        cosine_threshold: float = 0.2) -> Grade:
    context = "\n---\n".join(r.chunk.text for r in retrieved)
    try:
        user_content = GRADER_USER_TMPL.format(question=question, context=context)
        raw = provider.generate(
            [{"role": "system", "content": GRADER_SYSTEM},
             {"role": "user", "content": user_content}],
            json_mode=True, max_tokens=64,
        )
        parsed = json.loads(raw)
        return Grade(answerable=strict_bool(parsed["answerable"]),
                     confidence=int(parsed.get("confidence", 3)))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Any malformed field — including a non-boolean `answerable` strict_bool rejects — routes
        # to the cosine fallback, which fails toward refusal below threshold.
        return cosine_fallback_grade(retrieved, threshold=cosine_threshold)
    except Exception:
        logger.warning("grader provider call failed; using cosine fallback", exc_info=True)
        return cosine_fallback_grade(retrieved, threshold=cosine_threshold)
