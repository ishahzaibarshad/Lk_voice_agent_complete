import pytest

from config import load_provider_keys


def test_requires_tts_and_at_least_one_llm_or_stt_provider():
    with pytest.raises(RuntimeError) as error:
        load_provider_keys({})
    assert "OPENAI_API_KEY" in str(error.value)
    assert "GROQ_API_KEY" in str(error.value)
    assert "CARTESIA_API_KEY" in str(error.value)


def test_accepts_openai_and_cartesia():
    keys = load_provider_keys({"OPENAI_API_KEY": "openai-test", "CARTESIA_API_KEY": "cartesia-test"})
    assert keys.openai == "openai-test"
    assert keys.groq == ""


def test_accepts_groq_and_cartesia():
    keys = load_provider_keys({"GROQ_API_KEY": "groq-test", "CARTESIA_API_KEY": "cartesia-test"})
    assert keys.groq == "groq-test"
    assert keys.openai == ""


def test_strips_whitespace_from_configured_keys():
    keys = load_provider_keys({"GROQ_API_KEY": " groq-test ", "CARTESIA_API_KEY": " cartesia-test "})
    assert keys.groq == "groq-test"
    assert keys.cartesia == "cartesia-test"

