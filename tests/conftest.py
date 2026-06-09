"""Shared test fixtures. FakeModelProvider replaces the live OpenAI-compatible API so the
graph/pipeline/eval are tested deterministically without network or keys."""
import hashlib

import pytest


class FakeModelProvider:
    """Deterministic embed (hash-seeded 384-d vector) + scriptable generate.

    - embed(): stable per text, so retrieval order is reproducible.
    - generate(): returns canned_json when json_mode else canned_answer.
    """

    def __init__(self, canned_json: str = '{"answerable": true, "confidence": 5}',
                 canned_answer: str = "A grounded answer."):
        self.canned_json = canned_json
        self.canned_answer = canned_answer
        self.calls: list[dict] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [((h[i % len(h)] / 255.0) - 0.5) for i in range(384)]
            out.append(vec)
        return out

    def generate(self, messages, *, json_mode=False, max_tokens=512, temperature=0.0) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        return self.canned_json if json_mode else self.canned_answer


@pytest.fixture
def fake_provider():
    return FakeModelProvider()
