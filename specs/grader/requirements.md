# Grader Requirements

Source of truth for the grader prompt; keep in sync with `core/grader.py`.

## Prompts (verbatim)

GRADER_SYSTEM = "You are a strict grader. Reply ONLY with a JSON object."

GRADER_USER_TMPL = (
    "Question:\n{question}\n\n"
    "Retrieved context (the ONLY allowed source):\n{context}\n\n"
    "Decide if the question can be answered FROM THIS CONTEXT ALONE. "
    "Return exactly {\"answerable\": <true|false>, \"confidence\": <1-5 integer>}. "
    "answerable=false if the context does not contain the answer."
)
