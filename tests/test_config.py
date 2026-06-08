from genacademy_rag.config import PROVIDER_PRESETS, Settings


def test_provider_preset_resolves_base_url_key_and_model(monkeypatch):
    monkeypatch.setenv("GENACADEMY_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
    s = Settings.from_env()
    assert s.provider == "openrouter"
    assert s.gen_base_url == "https://openrouter.ai/api/v1"
    assert s.gen_api_key == "sk-test"
    assert s.gen_model == "meta-llama/llama-3.1-70b-instruct"


def test_known_presets_present():
    assert {"nebius", "openrouter", "openai", "gemma"} <= set(PROVIDER_PRESETS)
