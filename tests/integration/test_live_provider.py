import pytest

from genacademy_rag.config import Settings
from genacademy_rag.core.providers import OpenAICompatProvider


@pytest.mark.integration
def test_live_json_mode_returns_parseable_object():
    s = Settings.from_env()
    if not s.gen_api_key:
        pytest.skip("no generation key set")
    p = OpenAICompatProvider(s.gen_base_url, s.gen_api_key, s.gen_model)
    out = p.generate(
        [{"role": "system", "content": "Reply ONLY with JSON."},
         {"role": "user", "content": 'Return {"answerable": true, "confidence": 5}.'}],
        json_mode=True, max_tokens=64,
    )
    import json
    parsed = json.loads(out)
    assert "answerable" in parsed
