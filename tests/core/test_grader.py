from genacademy_rag.core.grader import cosine_fallback_grade, grade_answerability
from genacademy_rag.core.types import Chunk, Citation, RetrievedChunk
from tests.conftest import FakeModelProvider


def _rc(text, score=0.5):
    cit = Citation(doc_id="d1", title="t", source_type="github")
    chunk = Chunk(chunk_id="d1::0", doc_id="d1", ordinal=0, text=text, citation=cit)
    return RetrievedChunk(chunk=chunk, score=score)


def test_json_grader_parses_answerable_true():
    p = FakeModelProvider(canned_json='{"answerable": true, "confidence": 5}')
    g = grade_answerability("what is RAG?", [_rc("RAG retrieves then generates.")], p)
    assert g.answerable is True and g.confidence == 5


def test_json_grader_parses_refusal():
    p = FakeModelProvider(canned_json='{"answerable": false, "confidence": 1}')
    g = grade_answerability("who won the 2050 world cup?", [_rc("unrelated text")], p)
    assert g.answerable is False


def test_grader_falls_back_to_cosine_on_malformed_json():
    p = FakeModelProvider(canned_json="not json at all")
    g = grade_answerability("q", [_rc("x", score=0.9)], p, cosine_threshold=0.2)
    assert g.answerable is True            # fell back, top score 0.9 >= 0.2
    assert g.used_fallback is True


def test_cosine_fallback_refuses_when_below_threshold():
    g = cosine_fallback_grade([_rc("x", score=0.05)], threshold=0.2)
    assert g.answerable is False
