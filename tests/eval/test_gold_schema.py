import pytest

from genacademy_rag.eval.gold_schema import CATEGORIES, GoldQuestion, load_gold_set


def test_categories_cover_the_required_buckets():
    assert {"answerable", "exact_match", "chunking_stress", "multi_document",
            "ambiguous", "unanswerable"} == set(CATEGORIES)


def test_load_gold_set_parses_and_validates(tmp_path):
    yaml_text = """
- id: q1
  question: "What does the catalog say covers QLoRA?"
  category: exact_match
  answerable: true
  gold:
    - repo: awesome-agentic-ai-resources
      file_path: README.md
      commit_hash: 5dfb8691180dc4956107e86839998ba3a2ebd94f
      line_start: 40
      line_end: 41
- id: q2
  question: "What does the course say about week 8?"
  category: unanswerable
  answerable: false
  gold: []
"""
    p = tmp_path / "g.yaml"
    p.write_text(yaml_text)
    gold = load_gold_set(p)
    assert len(gold) == 2
    assert isinstance(gold[0], GoldQuestion)
    assert gold[0].category == "exact_match"
    assert gold[1].answerable is False and gold[1].gold == []


def test_answerable_question_must_have_gold_spans(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        '- id: q1\n  question: "x"\n  category: answerable\n  answerable: true\n  gold: []\n'
    )
    with pytest.raises(ValueError, match="answerable.*requires.*gold"):
        load_gold_set(bad)
