import ssl

import anyio
import httpx2
import pytest

from llm_client import (
    AnthropicClient,
    ChatMessage,
    ChatRequest,
    ErrorKind,
    LLMError,
    OpenAIClient,
    Provider,
    Role,
    Settings,
)
from tests.http_mocks import (
    RecordingTransport,
    anthropic_error,
    anthropic_message,
    anthropic_stream,
    anthropic_stream_error,
    anthropic_stream_events,
    http_client,
    interrupted_sse,
    json_response,
    openai_completion,
    openai_error,
    openai_stream,
    openai_stream_events,
)

CONVERSATION = ChatRequest(
    messages=[
        ChatMessage(role=Role.SYSTEM, content="Respondé en español."),
        ChatMessage(role=Role.USER, content="¿Qué es la entropía?"),
    ]
)


def openai_client(settings: Settings, transport: RecordingTransport) -> OpenAIClient:
    return OpenAIClient(settings.config_for(Provider.OPENAI), http_client=http_client(transport))


def anthropic_client(settings: Settings, transport: RecordingTransport) -> AnthropicClient:
    return AnthropicClient(
        settings.config_for(Provider.ANTHROPIC), http_client=http_client(transport)
    )


async def collect(client: OpenAIClient | AnthropicClient) -> list[str]:
    return [token async for token in client.stream(CONVERSATION)]


async def test_openai_generate_normalizes_the_completion(settings: Settings) -> None:
    transport = RecordingTransport([json_response(200, openai_completion("Es una medida."))])

    response = await openai_client(settings, transport).generate(CONVERSATION)

    assert response.ok
    assert response.provider is Provider.OPENAI
    assert response.content == "Es una medida."
    assert response.model == "gpt-4o-mini-2024-07-18"
    assert response.finish_reason == "stop"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (12, 30)
    assert response.latency_ms >= 0


async def test_openai_sends_messages_and_validated_parameters(settings: Settings) -> None:
    transport = RecordingTransport([json_response(200, openai_completion("ok"))])

    await openai_client(settings, transport).generate(CONVERSATION)

    sent = transport.sent_json()
    assert sent["model"] == "gpt-4o-mini"
    assert sent["messages"] == [
        {"role": "system", "content": "Respondé en español."},
        {"role": "user", "content": "¿Qué es la entropía?"},
    ]
    assert sent["max_completion_tokens"] == settings.llm_max_tokens
    assert sent["temperature"] == settings.openai_temperature
    assert transport.requests[-1].headers["authorization"] == "Bearer sk-test-openai"


async def test_openai_stream_yields_tokens(settings: Settings) -> None:
    transport = RecordingTransport([openai_stream(["La ", "entropía ", "mide."])])

    tokens = await collect(openai_client(settings, transport))

    assert tokens == ["La ", "entropía ", "mide."]
    assert transport.sent_json()["stream"] is True


async def test_anthropic_generate_moves_system_prompt_and_joins_text_blocks(
    settings: Settings,
) -> None:
    transport = RecordingTransport([json_response(200, anthropic_message("Es una medida."))])

    response = await anthropic_client(settings, transport).generate(CONVERSATION)

    sent = transport.sent_json()
    assert sent["system"] == "Respondé en español."
    assert sent["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]
    assert sent["output_config"] == {"effort": settings.anthropic_effort}
    assert "temperature" not in sent
    assert response.content == "Es una medida."
    assert response.finish_reason == "end_turn"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (15, 40)


async def test_anthropic_stream_yields_text_deltas(settings: Settings) -> None:
    transport = RecordingTransport([anthropic_stream(["La ", "entropía ", "mide."])])

    tokens = await collect(anthropic_client(settings, transport))

    assert tokens == ["La ", "entropía ", "mide."]


async def test_rate_limit_is_normalized_with_retry_after(settings: Settings) -> None:
    transport = RecordingTransport(
        [openai_error(429, "rate_limit_exceeded", headers={"retry-after": "7"})]
    )

    with pytest.raises(LLMError) as raised:
        await openai_client(settings, transport).generate(CONVERSATION)

    info = raised.value.info
    assert info.kind is ErrorKind.RATE_LIMIT
    assert info.retryable
    assert info.status_code == 429
    assert info.retry_after_seconds == 7
    assert len(transport.requests) == 1


async def test_exhausted_quota_is_not_retryable(settings: Settings) -> None:
    transport = RecordingTransport([openai_error(429, "insufficient_quota")])

    with pytest.raises(LLMError) as raised:
        await openai_client(settings, transport).generate(CONVERSATION)

    assert raised.value.info.kind is ErrorKind.QUOTA_EXCEEDED
    assert not raised.value.info.retryable


@pytest.mark.parametrize(
    ("status", "error_type", "kind"),
    [
        (401, "authentication_error", ErrorKind.AUTHENTICATION),
        (400, "invalid_request_error", ErrorKind.INVALID_REQUEST),
        (402, "billing_error", ErrorKind.QUOTA_EXCEEDED),
        (529, "overloaded_error", ErrorKind.PROVIDER_UNAVAILABLE),
    ],
)
async def test_anthropic_status_errors_are_classified(
    settings: Settings, status: int, error_type: str, kind: ErrorKind
) -> None:
    transport = RecordingTransport([anthropic_error(status, error_type)])

    with pytest.raises(LLMError) as raised:
        await anthropic_client(settings, transport).generate(CONVERSATION)

    assert raised.value.info.kind is kind
    assert raised.value.info.provider is Provider.ANTHROPIC


async def test_network_failure_becomes_a_connection_error(settings: Settings) -> None:
    transport = RecordingTransport([httpx2.ConnectError("sin red")])

    with pytest.raises(LLMError) as raised:
        await collect(anthropic_client(settings, transport))

    assert raised.value.info.kind is ErrorKind.CONNECTION
    assert raised.value.info.retryable


async def test_timeout_becomes_a_timeout_error(settings: Settings) -> None:
    transport = RecordingTransport([httpx2.ReadTimeout("lento")])

    with pytest.raises(LLMError) as raised:
        await openai_client(settings, transport).generate(CONVERSATION)

    assert raised.value.info.kind is ErrorKind.TIMEOUT


async def collect_until_failure(
    client: OpenAIClient | AnthropicClient,
) -> tuple[list[str], LLMError]:
    received: list[str] = []
    with pytest.raises(LLMError) as raised:
        async for token in client.stream(CONVERSATION):
            received.append(token)
    return received, raised.value


async def test_anthropic_stream_cut_mid_response_becomes_a_connection_error(
    settings: Settings,
) -> None:
    first_token_only = anthropic_stream_events(["La ", "entropía"])[:3]
    transport = RecordingTransport(
        [interrupted_sse(first_token_only, httpx2.ReadError("conexión reiniciada"))]
    )

    received, error = await collect_until_failure(anthropic_client(settings, transport))

    assert received == ["La "]
    assert error.info.kind is ErrorKind.CONNECTION
    assert error.info.provider is Provider.ANTHROPIC


async def test_anthropic_stream_timeout_before_first_event_becomes_a_timeout_error(
    settings: Settings,
) -> None:
    transport = RecordingTransport([interrupted_sse([], httpx2.ReadTimeout("lento"))])

    received, error = await collect_until_failure(anthropic_client(settings, transport))

    assert received == []
    assert error.info.kind is ErrorKind.TIMEOUT
    assert error.info.retryable


async def test_openai_stream_cut_mid_response_becomes_a_connection_error(
    settings: Settings,
) -> None:
    first_token_only = openai_stream_events(["La ", "entropía"])[:2]
    transport = RecordingTransport(
        [interrupted_sse(first_token_only, httpx2.ReadError("conexión reiniciada"))]
    )

    received, error = await collect_until_failure(openai_client(settings, transport))

    assert received == ["La "]
    assert error.info.kind is ErrorKind.CONNECTION


@pytest.mark.parametrize(
    ("error_type", "kind"),
    [
        ("overloaded_error", ErrorKind.PROVIDER_UNAVAILABLE),
        ("api_error", ErrorKind.PROVIDER_UNAVAILABLE),
        ("rate_limit_error", ErrorKind.RATE_LIMIT),
    ],
)
async def test_anthropic_error_event_inside_the_stream_is_classified_by_its_type(
    settings: Settings, error_type: str, kind: ErrorKind
) -> None:
    transport = RecordingTransport([anthropic_stream_error(error_type)])

    _, error = await collect_until_failure(anthropic_client(settings, transport))

    assert error.info.kind is kind
    assert error.info.retryable


@pytest.mark.parametrize(
    "network_failure",
    [
        ssl.SSLError("conexión TLS cerrada"),
        anyio.EndOfStream(),
        httpx2.DecodingError("respuesta corrupta"),
        httpx2.RemoteProtocolError("el servidor cortó la respuesta"),
    ],
    ids=["ssl", "end-of-stream", "decoding", "remote-protocol"],
)
async def test_any_network_failure_mid_stream_becomes_a_connection_error(
    settings: Settings, network_failure: Exception
) -> None:
    first_token_only = anthropic_stream_events(["La ", "entropía"])[:3]
    transport = RecordingTransport([interrupted_sse(first_token_only, network_failure)])

    received, error = await collect_until_failure(anthropic_client(settings, transport))

    assert received == ["La "]
    assert error.info.kind is ErrorKind.CONNECTION


async def test_error_event_inside_a_200_stream_reports_no_http_status(settings: Settings) -> None:
    transport = RecordingTransport([anthropic_stream_error("overloaded_error")])

    _, error = await collect_until_failure(anthropic_client(settings, transport))

    assert error.info.status_code is None
