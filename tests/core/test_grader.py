import logging

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


def test_json_grader_honors_stringified_false_as_refusal():
    # JSON-mode LLMs emit booleans as strings; bool("false") is True in Python, which would flip a
    # refusal into an answer (refusal bypass). strict_bool must read "false" as False — and because
    # the JSON parsed fine, this is NOT the cosine fallback path.
    p = FakeModelProvider(canned_json='{"answerable": "false", "confidence": 1}')
    g = grade_answerability("who won the 2050 world cup?", [_rc("unrelated", score=0.9)], p)
    assert g.answerable is False
    assert g.used_fallback is False


def test_json_grader_honors_stringified_true():
    p = FakeModelProvider(canned_json='{"answerable": "true", "confidence": 4}')
    g = grade_answerability("what is RAG?", [_rc("RAG retrieves then generates.")], p)
    assert g.answerable is True and g.used_fallback is False


def test_grader_falls_back_to_cosine_on_malformed_json():
    p = FakeModelProvider(canned_json="not json at all")
    g = grade_answerability("q", [_rc("x", score=0.9)], p, cosine_threshold=0.2)
    assert g.answerable is True            # fell back, top score 0.9 >= 0.2
    assert g.used_fallback is True


def test_grader_falls_back_when_answerable_is_not_a_real_boolean():
    # A non-boolean string (not "true"/"false") is unparseable → cosine fallback, which refuses
    # below threshold. Fail toward refusal, never answer on a garbage grade.
    p = FakeModelProvider(canned_json='{"answerable": "maybe", "confidence": 3}')
    g = grade_answerability("q", [_rc("x", score=0.05)], p, cosine_threshold=0.2)
    assert g.answerable is False and g.used_fallback is True


def test_grader_falls_back_to_cosine_on_provider_exception(caplog):
    class BrokenProvider:
        def generate(self, *args, **kwargs):
            raise TimeoutError("provider timed out")

    with caplog.at_level(logging.WARNING, logger="genacademy_rag.core.grader"):
        g = grade_answerability(
            "q",
            [_rc("x", score=0.05)],
            BrokenProvider(),
            cosine_threshold=0.2,
        )

    assert g.answerable is False
    assert g.used_fallback is True
    assert "grader provider call failed; using cosine fallback" in caplog.text


def test_cosine_fallback_refuses_when_below_threshold():
    g = cosine_fallback_grade([_rc("x", score=0.05)], threshold=0.2)
    assert g.answerable is False
