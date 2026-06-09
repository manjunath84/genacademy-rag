"""Strict coercion for JSON fields returned by JSON-mode LLMs. These models routinely emit
booleans as strings (`"false"`), and `bool("false")` is `True` in Python — so a naive
`bool(parsed[...])` silently flips a refusal into an answer (the load-bearing refusal path,
AGENTS.md §3) or a hallucination flag into "faithful". `strict_bool` honors a real `true`/`false`
(bool or string) and raises on anything else, so callers route the `ValueError` to a safe fallback
(the grader falls back to the cosine threshold; the eval disables the judge run-wide)."""
from __future__ import annotations


def strict_bool(value: object) -> bool:
    """Return the boolean meaning of a JSON value, raising on anything that isn't an unambiguous
    boolean. Accepts `True`/`False` and the strings `"true"`/`"false"` (case-insensitive, trimmed).
    Rejects ints, `"yes"`/`"1"`/`"maybe"`/`None`, etc. — fail toward the caller's safe path."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
    raise ValueError(f"expected a JSON boolean, got {value!r}")
