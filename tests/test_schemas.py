import pytest
from pydantic import SecretStr, ValidationError

from llm_client import (
    ChatMessage,
    ChatRequest,
    ErrorInfo,
    ErrorKind,
    LLMConfig,
    ModelResponse,
    Provider,
    RetryPolicy,
    Role,
)


def openai_config(**overrides: object) -> LLMConfig:
    fields: dict[str, object] = {
        "provider": Provider.OPENAI,
        "model": "gpt-4o-mini",
        "api_key": SecretStr("sk-secret-value"),
    }
    return LLMConfig(**(fields | overrides))


def test_chat_message_rejects_blank_content() -> None:
    with pytest.raises(ValidationError, match="vacío"):
        ChatMessage(role=Role.USER, content="   ")


def test_chat_message_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="hola")


def test_chat_request_from_plain_prompt_is_a_single_user_message() -> None:
    request = ChatRequest.from_prompt("¿Qué es la entropía?")

    assert request.messages == [ChatMessage(role=Role.USER, content="¿Qué es la entropía?")]


def test_chat_request_must_end_with_a_user_message() -> None:
    with pytest.raises(ValidationError, match="usuario"):
        ChatRequest(
            messages=[
                ChatMessage(role=Role.USER, content="Hola"),
                ChatMessage(role=Role.ASSISTANT, content="¡Hola!"),
            ]
        )


def test_chat_request_needs_at_least_one_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 2.5),
        ("temperature", -0.1),
        ("max_tokens", 0),
        ("max_tokens", 200_000),
        ("timeout_seconds", 0),
        ("model", ""),
    ],
)
def test_llm_config_rejects_values_out_of_range(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        openai_config(**{field: value})


def test_llm_config_rejects_blank_api_key() -> None:
    with pytest.raises(ValidationError, match="API key"):
        openai_config(api_key=SecretStr("  "))


def test_llm_config_rejects_temperature_for_anthropic() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        LLMConfig(
            provider=Provider.ANTHROPIC,
            model="claude-opus-5-5",
            api_key=SecretStr("sk-ant-test"),
            temperature=0.5,
        )


def test_llm_config_rejects_effort_for_openai() -> None:
    with pytest.raises(ValidationError, match="effort"):
        openai_config(effort="low")


def test_llm_config_rejects_unknown_effort() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(
            provider=Provider.ANTHROPIC,
            model="claude-opus-5-5",
            api_key=SecretStr("sk-ant-test"),
            effort="turbo",
        )


def test_llm_config_never_exposes_the_api_key() -> None:
    config = openai_config()

    assert "sk-secret-value" not in repr(config)
    assert "sk-secret-value" not in str(config)
    assert "sk-secret-value" not in config.model_dump_json()
    assert config.api_key.get_secret_value() == "sk-secret-value"


def test_retry_policy_rejects_base_delay_above_max_delay() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(base_delay_seconds=10, max_delay_seconds=5)


def test_retry_policy_backoff_grows_exponentially_within_jitter_bounds() -> None:
    policy = RetryPolicy(max_attempts=5, base_delay_seconds=1, max_delay_seconds=30)

    for attempt, ceiling in [(1, 1), (2, 2), (3, 4), (4, 8)]:
        delay = policy.delay_for(attempt)
        assert ceiling / 2 <= delay <= ceiling


def test_retry_policy_waits_what_retry_after_asks() -> None:
    policy = RetryPolicy(base_delay_seconds=1, max_delay_seconds=20)

    assert policy.delay_for(1, retry_after_seconds=7) == 7


def test_retry_policy_gives_up_when_retry_after_exceeds_the_max_delay() -> None:
    policy = RetryPolicy(base_delay_seconds=1, max_delay_seconds=20)

    assert policy.delay_for(1, retry_after_seconds=90) is None


def test_error_info_reports_which_errors_are_retryable() -> None:
    def info(kind: ErrorKind) -> ErrorInfo:
        return ErrorInfo(kind=kind, provider=Provider.OPENAI, message="x")

    retryable = {kind for kind in ErrorKind if info(kind).retryable}

    assert retryable == {
        ErrorKind.RATE_LIMIT,
        ErrorKind.TIMEOUT,
        ErrorKind.CONNECTION,
        ErrorKind.PROVIDER_UNAVAILABLE,
    }
    assert info(ErrorKind.RATE_LIMIT).model_dump()["retryable"] is True


def test_model_response_ok_reflects_error() -> None:
    success = ModelResponse(provider=Provider.OPENAI, model="gpt-4o-mini", content="hola")
    failure = ModelResponse(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        error=ErrorInfo(kind=ErrorKind.TIMEOUT, provider=Provider.OPENAI, message="timeout"),
    )

    assert success.ok
    assert not failure.ok
