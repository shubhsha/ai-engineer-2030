from pathlib import Path

import pytest

from ai_engineer.foundations.settings import (
    Settings,
    SettingsError,
    parse_env_file,
)


def test_defaults_are_usable():
    settings = Settings.from_env({})
    assert settings.model_config == Path("configs/models.toml")
    assert settings.log_level == "INFO"
    assert settings.api_key is None


def test_every_problem_is_reported_at_once():
    with pytest.raises(SettingsError) as err:
        Settings.from_env(
            {"AIE_LOG_LEVEL": "loud", "AIE_MODEL_CONFIG": "models.json"}
        )
    assert len(err.value.problems) == 2


def test_api_key_never_appears_in_repr():
    settings = Settings.from_env({"ANTHROPIC_API_KEY": "sk-secret"})
    assert settings.api_key == "sk-secret"
    assert "sk-secret" not in repr(settings)


def test_parse_env_file():
    text = "# comment\n\nA=1\nB = 'two'\nC=\"three\"\n"
    assert parse_env_file(text) == {"A": "1", "B": "two", "C": "three"}
    with pytest.raises(ValueError, match="line 1"):
        parse_env_file("not a pair")
