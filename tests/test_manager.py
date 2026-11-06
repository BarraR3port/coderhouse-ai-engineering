from collections.abc import AsyncGenerator, Iterable
from typing import override

import httpx2
import pytest

from llm_client import (
    AnthropicClient,
    AsyncLLMManager,
    BaseLLMClient,
    ChatRequest,
    ConfigurationError,
    ErrorInfo,
    ErrorKind,
    LLMConfig,
    LLMError,
    ModelResponse,
    OpenAIClient,
    Provider,
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
)

type GenerateOutcome = ModelResponse | ErrorKind
type StreamOutcome = list[str | ErrorKind]


def failure(kind: ErrorKind, provider: Provider = Provider.OPENAI) -> LLMError:
    return LLMError(ErrorInfo(kind=kind, provider=provider, message=f"simulated {kind}"))


class ScriptedClient(BaseLLMClient):
    def __init__(
        self,
        config: LLMConfig,
        generate_outcomes: Iterable[GenerateOutcome] = (),
        stream_outcomes: Iterable[StreamOutcome] = (),
    ) -> None:
        super().__init__(config)
        self._generate_outcomes = iter(generate_outcomes)
        self._stream_outcomes = iter(stream_outcomes)
        self.requests: list[ChatRequest] = []
        self.closed = False
        self.open_streams = 0

    @override
    async def generate(self, request: ChatRequest) -> ModelResponse:
        self.requests.append(request)
        outcome = next(self._generate_outcomes)
        if isinstance(outcome, ErrorKind):
            raise failure(outcome, self.provider)
        return outcome

    @override
    async def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        self.requests.append(request)
        self.open_streams += 1
        try:
            for item in next(self._stream_outcomes):
                if isinstance(item, ErrorKind):
                    raise failure(item, self.provider)
                yield item
        finally:
            self.open_streams -= 1

    @override
    async def aclose(self) -> None:
        self.closed = True


class RecordedSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def answer(content: str, provider: Provider = Provider.OPENAI) -> ModelResponse:
    return ModelResponse(provider=provider, model="test-model", content=content)


def manager_with(
    settings: Settings, **script: Iterable
) -> tuple[AsyncLLMManager, dict[Provider, ScriptedClient], RecordedSleep]:
    clients: dict[Provider, ScriptedClient] = {}
    sleep = RecordedSleep()

    def build(config: LLMConfig) -> ScriptedClient:
        clients[config.provider] = ScriptedClient(config, **script)
        return clients[config.provider]

    return AsyncLLMManager(settings, client_builder=build, sleep=sleep), clients, sleep


async def consume(manager: AsyncLLMManager, **kwargs: object) -> list[str]:
    return [token async for token in manager.stream("¿Qué es la entropía?", **kwargs)]


async def test_generate_uses_the_default_provider_and_accepts_a_plain_prompt(
    settings: Settings,
) -> None:
    manager, clients, _ = manager_with(settings, generate_outcomes=[answer("hola")])

    response = await manager.generate("¿Qué es la entropía?")

    assert response.content == "hola"
    assert response.attempts == 1
    assert list(clients) == [Provider.OPENAI]
    assert clients[Provider.OPENAI].requests[0].messages[0].content == "¿Qué es la entropía?"


async def test_provider_can_be_switched_per_call(settings: Settings) -> None:
    manager, clients, _ = manager_with(
        settings, generate_outcomes=[answer("desde claude", Provider.ANTHROPIC)]
    )

    response = await manager.generate("hola", provider=Provider.ANTHROPIC)

    assert response.provider is Provider.ANTHROPIC
    assert isinstance(manager.client(Provider.ANTHROPIC), ScriptedClient)
    assert clients[Provider.ANTHROPIC].config.model == settings.anthropic_model


async def test_clients_are_created_once_per_provider(settings: Settings) -> None:
    manager, _, _ = manager_with(settings)

    assert manager.client(Provider.OPENAI) is manager.client(Provider.OPENAI)


async def test_generate_retries_rate_limits_with_backoff_then_succeeds(settings: Settings) -> None:
    manager, _, sleep = manager_with(
        settings, generate_outcomes=[ErrorKind.RATE_LIMIT, ErrorKind.CONNECTION, answer("ok")]
    )

    response = await manager.generate("hola")

    assert response.ok
    assert response.attempts == 3
    assert len(sleep.delays) == 2


async def test_generate_returns_a_structured_error_after_exhausting_retries(
    settings: Settings,
) -> None:
    manager, _, sleep = manager_with(settings, generate_outcomes=[ErrorKind.RATE_LIMIT] * 3)

    response = await manager.generate("hola")

    assert not response.ok
    assert response.error.kind is ErrorKind.RATE_LIMIT
    assert response.error.attempts == 3
    assert response.attempts == 3
    assert len(sleep.delays) == 2


async def test_generate_does_not_retry_errors_that_cannot_recover(settings: Settings) -> None:
    manager, clients, sleep = manager_with(settings, generate_outcomes=[ErrorKind.AUTHENTICATION])

    response = await manager.generate("hola")

    assert response.error.kind is ErrorKind.AUTHENTICATION
    assert len(clients[Provider.OPENAI].requests) == 1
    assert sleep.delays == []


async def test_invalid_prompt_is_rejected_before_calling_the_provider(settings: Settings) -> None:
    manager, clients, _ = manager_with(settings)

    with pytest.raises(ValueError, match="vacío"):
        await manager.generate("   ")

    assert clients == {}


async def test_missing_key_is_a_configuration_error(settings: Settings) -> None:
    manager, _, _ = manager_with(Settings(_env_file=None, openai_api_key="sk-test"))

    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        await manager.generate("hola", provider=Provider.ANTHROPIC)


async def test_stream_yields_tokens(settings: Settings) -> None:
    manager, _, _ = manager_with(settings, stream_outcomes=[["La ", "entropía"]])

    assert await consume(manager) == ["La ", "entropía"]


async def test_stream_retries_failures_before_the_first_token(settings: Settings) -> None:
    manager, _, sleep = manager_with(
        settings, stream_outcomes=[[ErrorKind.TIMEOUT], ["La ", "entropía"]]
    )

    assert await consume(manager) == ["La ", "entropía"]
    assert len(sleep.delays) == 1


async def test_stream_does_not_retry_after_partial_output(settings: Settings) -> None:
    manager, _, sleep = manager_with(
        settings, stream_outcomes=[["La ", ErrorKind.CONNECTION], ["no debería llegar"]]
    )
    received: list[str] = []

    with pytest.raises(LLMError) as raised:
        async for token in manager.stream("hola"):
            received.append(token)

    assert received == ["La "]
    assert raised.value.info.kind is ErrorKind.CONNECTION
    assert sleep.delays == []


async def test_stream_raises_a_structured_error_after_exhausting_retries(
    settings: Settings,
) -> None:
    manager, _, _ = manager_with(settings, stream_outcomes=[[ErrorKind.RATE_LIMIT]] * 3)

    with pytest.raises(LLMError) as raised:
        await consume(manager)

    assert raised.value.info.kind is ErrorKind.RATE_LIMIT
    assert raised.value.info.attempts == 3


async def test_closing_the_manager_closes_every_client(settings: Settings) -> None:
    manager, clients, _ = manager_with(settings)

    async with manager:
        manager.client(Provider.OPENAI)
        manager.client(Provider.ANTHROPIC)

    assert all(client.closed for client in clients.values())


async def test_manager_retries_a_real_sdk_rate_limit(settings: Settings) -> None:
    transport = RecordingTransport(
        [
            openai_error(429, "rate_limit_exceeded", headers={"retry-after": "2"}),
            json_response(200, openai_completion("Es una medida del desorden.")),
        ]
    )
    sleep = RecordedSleep()
    manager = AsyncLLMManager(
        settings,
        client_builder=lambda config: OpenAIClient(config, http_client=http_client(transport)),
        sleep=sleep,
    )

    response = await manager.generate("¿Qué es la entropía?")

    assert response.content == "Es una medida del desorden."
    assert response.attempts == 2
    assert sleep.delays == [2]
    assert len(transport.requests) == 2


def manager_over_http(
    settings: Settings, client_class: type[BaseLLMClient], responses: list
) -> tuple[AsyncLLMManager, RecordingTransport, RecordedSleep]:
    transport = RecordingTransport(responses)
    sleep = RecordedSleep()
    manager = AsyncLLMManager(
        settings,
        client_builder=lambda config: client_class(config, http_client=http_client(transport)),
        sleep=sleep,
    )
    return manager, transport, sleep


async def test_generate_retries_an_overloaded_provider(settings: Settings) -> None:
    manager, transport, sleep = manager_over_http(
        settings,
        AnthropicClient,
        [anthropic_error(529, "overloaded_error"), json_response(200, anthropic_message("ok"))],
    )

    response = await manager.generate("hola", provider=Provider.ANTHROPIC)

    assert response.content == "ok"
    assert response.attempts == 2
    assert len(sleep.delays) == 1
    assert len(transport.requests) == 2


async def test_stream_retries_an_error_event_sent_before_the_first_token(
    settings: Settings,
) -> None:
    manager, transport, _ = manager_over_http(
        settings,
        AnthropicClient,
        [anthropic_stream_error("overloaded_error"), anthropic_stream(["La ", "entropía"])],
    )

    tokens = [token async for token in manager.stream("hola", provider=Provider.ANTHROPIC)]

    assert tokens == ["La ", "entropía"]
    assert len(transport.requests) == 2


async def test_stream_retries_a_network_timeout_before_the_first_token(
    settings: Settings,
) -> None:
    manager, transport, _ = manager_over_http(
        settings,
        AnthropicClient,
        [interrupted_sse([], httpx2.ReadTimeout("lento")), anthropic_stream(["ok"])],
    )

    tokens = [token async for token in manager.stream("hola", provider=Provider.ANTHROPIC)]

    assert tokens == ["ok"]
    assert len(transport.requests) == 2


async def test_stream_cut_after_the_first_token_raises_without_retrying(
    settings: Settings,
) -> None:
    first_token_only = anthropic_stream_events(["La ", "entropía"])[:3]
    manager, transport, _ = manager_over_http(
        settings,
        AnthropicClient,
        [interrupted_sse(first_token_only, httpx2.ReadError("conexión reiniciada"))],
    )
    received: list[str] = []

    with pytest.raises(LLMError) as raised:
        async for token in manager.stream("hola", provider=Provider.ANTHROPIC):
            received.append(token)

    assert received == ["La "]
    assert raised.value.info.kind is ErrorKind.CONNECTION
    assert len(transport.requests) == 1


async def test_retry_after_longer_than_the_max_delay_is_reported_instead_of_waited(
    settings: Settings,
) -> None:
    manager, transport, sleep = manager_over_http(
        settings,
        OpenAIClient,
        [openai_error(429, "rate_limit_exceeded", headers={"retry-after": "120"})],
    )

    response = await manager.generate("hola")

    assert response.error.kind is ErrorKind.RATE_LIMIT
    assert response.error.retry_after_seconds == 120
    assert sleep.delays == []
    assert len(transport.requests) == 1


async def test_closing_the_stream_early_closes_the_provider_stream(settings: Settings) -> None:
    manager, clients, _ = manager_with(settings, stream_outcomes=[["La ", "entropía", "mide"]])
    tokens = manager.stream("hola")

    assert await anext(tokens) == "La "
    await tokens.aclose()

    assert clients[Provider.OPENAI].open_streams == 0
