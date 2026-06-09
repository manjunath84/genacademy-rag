from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk
from genacademy_rag.eval.faithfulness_eval import (
    FAITHFULNESS_JUDGE_SYSTEM,
    citation_grounding_score,
    llm_judge_score,
)
from tests.conftest import FakeModelProvider


def _rc(text):
    cit = Citation(doc_id="d1", title="t", source_type="github")
    return RetrievedChunk(
        chunk=Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0, text=text, citation=cit),
        score=1.0,
    )


def test_citation_grounding_true_when_answer_overlaps_chunks():
    retrieved = [_rc("RAG retrieves documents then generates an answer.")]
    assert citation_grounding_score("RAG retrieves documents", retrieved) is True


def test_citation_grounding_false_when_answer_is_fabricated():
    retrieved = [_rc("This text is entirely about cooking pasta.")]
    assert citation_grounding_score("The capital of France is Paris.", retrieved) is False


def test_empty_or_all_stopword_answer_is_not_faithful():
    # An answer with no content words can't be grounded — that's a generation failure, not a
    # faithful answer. Scoring it True would silently inflate the graded faithfulness %.
    assert citation_grounding_score("", [_rc("anything relevant")]) is False
    assert citation_grounding_score("the a of to is", [_rc("anything relevant")]) is False


def test_llm_judge_parses_pinned_json():
    p = FakeModelProvider(canned_json='{"faithful": true, "hallucinated_claims": [], "score": 5}')
    out = llm_judge_score("q", "a", [_rc("ctx")], p)
    assert out["faithful"] is True and out["score"] == 5
    assert "ONLY" in FAITHFULNESS_JUDGE_SYSTEM  # verbatim pinned prompt present


def test_llm_judge_honors_stringified_false():
    # Same string-boolean hazard as the grader: a judge that flags a hallucination as "false" must
    # not be coerced to faithful=True and silently inflate the score.
    p = FakeModelProvider(
        canned_json='{"faithful": "false", "hallucinated_claims": ["x"], "score": 1}')
    out = llm_judge_score("q", "a", [_rc("ctx")], p)
    assert out["faithful"] is False
