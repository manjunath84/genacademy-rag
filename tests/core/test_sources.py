"""Tests for the pure answer-card presentation helpers (core/sources.py)."""

from genacademy_rag.core.sources import confidence_bucket


def test_confidence_bucket_boundaries():
    assert confidence_bucket(1) == "low"
    assert confidence_bucket(2) == "low"
    assert confidence_bucket(3) == "medium"
    assert confidence_bucket(4) == "high"
    assert confidence_bucket(5) == "high"
