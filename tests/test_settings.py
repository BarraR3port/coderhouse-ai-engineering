from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_client import ConfigurationError, Provider, Settings


def test_settings_load_from_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    monkeypatch.setenv("ANTHROPIC_EFFORT", "medium")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")

    settings = Settings(_env_file=None)
    config = settings.config_for(Provider.ANTHROPIC)

    assert settings.llm_provider is Provider.ANTHROPIC
    assert settings.configured_providers() == [Provider.ANTHROPIC]
    assert config.api_key.get_secret_value() == "sk-ant-env"
    assert config.effort == "medium"
    assert config.max_tokens == 2048
    assert config.temperature is None


def test_settings_load_from_env_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\nOPENAI_TEMPERATURE=0.2\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)
    config = settings.config_for(Provider.OPENAI)

    assert config.api_key.get_secret_value() == "sk-from-file"
    assert config.temperature == 0.2
    assert config.effort is None


def test_missing_key_raises_configuration_error_naming_the_variable() -> None:
    settings = Settings(_env_file=None, openai_api_key="sk-test")

    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        settings.config_for(Provider.ANTHROPIC)


def test_blank_key_counts_as_missing() -> None:
    settings = Settings(_env_file=None, openai_api_key="   ")

    assert settings.configured_providers() == []


def test_settings_reject_out_of_range_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_TEMPERATURE", "3")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_build_the_retry_policy() -> None:
    settings = Settings(_env_file=None, llm_max_attempts=5, llm_base_delay_seconds=0.5)

    policy = settings.retry_policy()

    assert policy.max_attempts == 5
    assert policy.base_delay_seconds == 0.5


@pytest.mark.parametrize(
    ("variable", "value"),
    [("OPENAI_MODEL", ""), ("ANTHROPIC_MODEL", "  "), ("LLM_BASE_DELAY_SECONDS", "30")],
)
def test_settings_reject_invalid_values_at_startup(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv("LLM_MAX_DELAY_SECONDS", "20")
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_env_file_is_read_from_the_project_root_whatever_the_working_directory() -> None:
    env_file = Path(Settings.model_config["env_file"])

    assert env_file.is_absolute()
    assert env_file == Path(__file__).resolve().parent.parent / ".env"
