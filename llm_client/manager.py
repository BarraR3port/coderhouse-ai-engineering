import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from types import TracebackType
from typing import Self

from llm_client.clients import AnthropicClient, BaseLLMClient, OpenAIClient
from llm_client.errors import LLMError
from llm_client.schemas import ChatRequest, ErrorInfo, LLMConfig, ModelResponse, Prompt, Provider
from llm_client.settings import Settings

logger = logging.getLogger(__name__)

type ClientBuilder = Callable[[LLMConfig], BaseLLMClient]
type Sleep = Callable[[float], Awaitable[None]]


def build_client(config: LLMConfig) -> BaseLLMClient:
    match config.provider:
        case Provider.OPENAI:
            return OpenAIClient(config)
        case Provider.ANTHROPIC:
            return AnthropicClient(config)


class AsyncLLMManager:
    def __init__(
        self,
        settings: Settings,
        *,
        client_builder: ClientBuilder = build_client,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._retry = settings.retry_policy()
        self._client_builder = client_builder
        self._sleep = sleep
        self._clients: dict[Provider, BaseLLMClient] = {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    def client(self, provider: Provider | None = None) -> BaseLLMClient:
        provider = provider or self._settings.llm_provider
        if provider not in self._clients:
            self._clients[provider] = self._client_builder(self._settings.config_for(provider))
        return self._clients[provider]

    async def generate(self, prompt: Prompt, *, provider: Provider | None = None) -> ModelResponse:
        request = ChatRequest.from_prompt(prompt)
        client = self.client(provider)
        attempt = 1
        while True:
            try:
                response = await client.generate(request)
                return response.model_copy(update={"attempts": attempt})
            except LLMError as error:
                info = error.info.model_copy(update={"attempts": attempt})
                if not await self._backoff_before_retry(info, attempt):
                    logger.warning("generate() falló sin más reintentos: %s", info.message)
                    return ModelResponse(
                        provider=client.provider,
                        model=client.config.model,
                        attempts=attempt,
                        error=info,
                    )
                attempt += 1

    async def stream(
        self, prompt: Prompt, *, provider: Provider | None = None
    ) -> AsyncGenerator[str, None]:
        request = ChatRequest.from_prompt(prompt)
        client = self.client(provider)
        attempt = 1
        while True:
            started_output = False
            try:
                async with aclosing(client.stream(request)) as tokens:
                    async for token in tokens:
                        started_output = True
                        yield token
                return
            except LLMError as error:
                info = error.info.model_copy(update={"attempts": attempt})
                if started_output or not await self._backoff_before_retry(info, attempt):
                    raise LLMError(info) from error
                attempt += 1

    async def aclose(self) -> None:
        clients, self._clients = list(self._clients.values()), {}
        await asyncio.gather(*(client.aclose() for client in clients))

    async def _backoff_before_retry(self, info: ErrorInfo, attempt: int) -> bool:
        if not info.retryable or attempt >= self._retry.max_attempts:
            return False
        delay = self._retry.delay_for(attempt, info.retry_after_seconds)
        if delay is None:
            return False
        logger.info(
            "%s respondió %s (intento %d/%d); reintento en %.1f s",
            info.provider,
            info.kind,
            attempt,
            self._retry.max_attempts,
            delay,
        )
        await self._sleep(delay)
        return True
