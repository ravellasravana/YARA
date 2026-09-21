"""Settings must accept API keys both as constructor kwargs and from env."""

import pytest

from yara.config import Settings

KEY_FIELDS = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # A developer's real key must not leak into (or mask) these assertions.
    for env in KEY_FIELDS.values():
        monkeypatch.delenv(env, raising=False)


@pytest.mark.parametrize("field", sorted(KEY_FIELDS))
def test_key_accepted_by_field_name(field):
    assert getattr(Settings(**{field: "sk-test"}), field) == "sk-test"


@pytest.mark.parametrize("field", sorted(KEY_FIELDS))
def test_key_accepted_by_alias(field):
    assert getattr(Settings(**{KEY_FIELDS[field]: "sk-test"}), field) == "sk-test"


@pytest.mark.parametrize("field", sorted(KEY_FIELDS))
def test_key_read_from_unprefixed_env_var(field, monkeypatch):
    monkeypatch.setenv(KEY_FIELDS[field], "sk-env")
    assert getattr(Settings(), field) == "sk-env"


def test_explicit_kwarg_beats_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert Settings(anthropic_api_key="sk-arg").anthropic_api_key == "sk-arg"


def test_keys_default_to_none():
    s = Settings()
    assert all(getattr(s, f) is None for f in KEY_FIELDS)
