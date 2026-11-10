from collections.abc import Iterator

import pytest

from llm_client import Settings
from tests.fake_api import FakeAPI, running_fake_api

PROVIDER_ENV_VARS = (
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_TEMPERATURE",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_EFFORT",
    "LLM_MAX_TOKENS",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_ATTEMPTS",
    "LLM_BASE_DELAY_SECONDS",
    "LLM_MAX_DELAY_SECONDS",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test-openai",
        anthropic_api_key="sk-ant-test",
        llm_max_attempts=3,
    )


@pytest.fixture
def fast_retry_settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test-openai",
        anthropic_api_key="sk-ant-test",
        llm_max_attempts=3,
        llm_base_delay_seconds=0.01,
        llm_max_delay_seconds=0.02,
    )


@pytest.fixture
def fake_api() -> Iterator[FakeAPI]:
    with running_fake_api() as api:
        yield api
