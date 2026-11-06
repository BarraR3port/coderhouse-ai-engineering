from pathlib import Path
from typing import Self

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_client.errors import ConfigurationError
from llm_client.schemas import (
    Attempts,
    DelaySeconds,
    Effort,
    LLMConfig,
    MaxTokens,
    ModelName,
    Provider,
    RetryPolicy,
    Temperature,
    TimeoutSeconds,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: Provider = Provider.OPENAI

    openai_api_key: SecretStr | None = None
    openai_model: ModelName = "gpt-4o-mini"
    openai_temperature: Temperature = 0.7

    anthropic_api_key: SecretStr | None = None
    anthropic_model: ModelName = "claude-opus-5-5"
    anthropic_effort: Effort = "low"

    llm_max_tokens: MaxTokens = 16_000
    llm_timeout_seconds: TimeoutSeconds = 60.0
    llm_max_attempts: Attempts = 3
    llm_base_delay_seconds: DelaySeconds = 1.0
    llm_max_delay_seconds: DelaySeconds = 20.0

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def blank_key_means_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def base_delay_within_max_delay(self) -> Self:
        if self.llm_base_delay_seconds > self.llm_max_delay_seconds:
            raise ValueError("LLM_BASE_DELAY_SECONDS no puede superar a LLM_MAX_DELAY_SECONDS")
        return self

    def configured_providers(self) -> list[Provider]:
        return [provider for provider in Provider if self._api_key(provider) is not None]

    def config_for(self, provider: Provider) -> LLMConfig:
        api_key = self._api_key(provider)
        if api_key is None:
            raise ConfigurationError(
                f"Falta {provider.upper()}_API_KEY: copiá .env.example a .env y completala"
            )
        shared = {
            "provider": provider,
            "api_key": api_key,
            "max_tokens": self.llm_max_tokens,
            "timeout_seconds": self.llm_timeout_seconds,
        }
        match provider:
            case Provider.OPENAI:
                return LLMConfig(
                    **shared, model=self.openai_model, temperature=self.openai_temperature
                )
            case Provider.ANTHROPIC:
                return LLMConfig(**shared, model=self.anthropic_model, effort=self.anthropic_effort)

    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=self.llm_max_attempts,
            base_delay_seconds=self.llm_base_delay_seconds,
            max_delay_seconds=self.llm_max_delay_seconds,
        )

    def _api_key(self, provider: Provider) -> SecretStr | None:
        match provider:
            case Provider.OPENAI:
                return self.openai_api_key
            case Provider.ANTHROPIC:
                return self.anthropic_api_key
