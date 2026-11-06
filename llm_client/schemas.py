import random
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


type Effort = Literal["low", "medium", "high", "xhigh", "max"]

ModelName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Temperature = Annotated[float, Field(ge=0.0, le=2.0)]
MaxTokens = Annotated[int, Field(ge=1, le=128_000)]
TimeoutSeconds = Annotated[float, Field(gt=0, le=600)]
Attempts = Annotated[int, Field(ge=1, le=10)]
DelaySeconds = Annotated[float, Field(gt=0, le=120)]


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(max_length=200_000)

    @field_validator("content")
    @classmethod
    def content_is_not_blank(cls, content: str) -> str:
        if not content.strip():
            raise ValueError("el contenido del mensaje no puede estar vacío")
        return content


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: list[ChatMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def ends_with_user_turn(self) -> Self:
        if self.messages[-1].role is not Role.USER:
            raise ValueError("el último mensaje de la conversación debe ser del usuario")
        return self

    @classmethod
    def from_prompt(cls, prompt: "Prompt") -> Self:
        if isinstance(prompt, ChatRequest):
            return cls.model_validate(prompt.model_dump())
        if isinstance(prompt, str):
            return cls(messages=[ChatMessage(role=Role.USER, content=prompt)])
        return cls(messages=list(prompt))


type Prompt = str | Sequence[ChatMessage] | ChatRequest


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    model: ModelName
    api_key: SecretStr
    max_tokens: MaxTokens = 16_000
    timeout_seconds: TimeoutSeconds = 60.0
    temperature: Temperature | None = None
    effort: Effort | None = None

    @field_validator("api_key")
    @classmethod
    def api_key_is_not_blank(cls, api_key: SecretStr) -> SecretStr:
        if not api_key.get_secret_value().strip():
            raise ValueError("la API key no puede estar vacía")
        return api_key

    @model_validator(mode="after")
    def parameters_match_provider(self) -> Self:
        if self.provider is Provider.ANTHROPIC and self.temperature is not None:
            raise ValueError("Anthropic no acepta temperature desde la API 2026-10-21: usá effort")
        if self.provider is Provider.OPENAI and self.effort is not None:
            raise ValueError("effort solo aplica a Anthropic: para OpenAI usá temperature")
        return self


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_attempts: Attempts = 3
    base_delay_seconds: DelaySeconds = 1.0
    max_delay_seconds: DelaySeconds = 20.0

    @model_validator(mode="after")
    def base_delay_within_max(self) -> Self:
        if self.base_delay_seconds > self.max_delay_seconds:
            raise ValueError("base_delay_seconds no puede superar a max_delay_seconds")
        return self

    def delay_for(self, attempt: int, retry_after_seconds: float | None = None) -> float | None:
        if retry_after_seconds is not None:
            return retry_after_seconds if retry_after_seconds <= self.max_delay_seconds else None
        backoff = min(self.base_delay_seconds * 2 ** (attempt - 1), self.max_delay_seconds)
        return random.uniform(backoff / 2, backoff)


class ErrorKind(StrEnum):
    AUTHENTICATION = "authentication"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN = "unknown"


RETRYABLE_ERRORS = frozenset(
    {
        ErrorKind.RATE_LIMIT,
        ErrorKind.TIMEOUT,
        ErrorKind.CONNECTION,
        ErrorKind.PROVIDER_UNAVAILABLE,
    }
)


class ErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ErrorKind
    provider: Provider
    message: str
    status_code: int | None = None
    retry_after_seconds: float | None = Field(default=None, ge=0)
    attempts: int = Field(default=1, ge=1)

    @computed_field
    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_ERRORS


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    latency_ms: float = Field(default=0.0, ge=0)
    attempts: int = Field(default=1, ge=1)
    error: ErrorInfo | None = None

    @computed_field
    @property
    def ok(self) -> bool:
        return self.error is None
