from collections.abc import AsyncGenerator
from time import perf_counter
from typing import Any, override

import httpx2
from anthropic import AsyncAnthropic

from llm_client.clients.base import BaseLLMClient, elapsed_ms
from llm_client.errors import PROVIDER_FAILURES, LLMError, classify_sdk_error
from llm_client.schemas import ChatRequest, LLMConfig, ModelResponse, Role, Usage


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: LLMConfig, http_client: httpx2.AsyncClient | None = None) -> None:
        super().__init__(config)
        self._sdk = AsyncAnthropic(
            api_key=config.api_key.get_secret_value(),
            timeout=config.timeout_seconds,
            max_retries=0,
            http_client=http_client,
        )

    @override
    async def generate(self, request: ChatRequest) -> ModelResponse:
        started = perf_counter()
        try:
            message = await self._sdk.messages.create(**self._payload(request))
        except PROVIDER_FAILURES as error:
            raise LLMError(classify_sdk_error(error, self.provider)) from error

        return ModelResponse(
            provider=self.provider,
            model=message.model,
            content="".join(block.text for block in message.content if block.type == "text"),
            finish_reason=message.stop_reason,
            usage=Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
            latency_ms=elapsed_ms(started),
        )

    @override
    async def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        try:
            async with self._sdk.messages.stream(**self._payload(request)) as events:
                async for token in events.text_stream:
                    yield token
        except PROVIDER_FAILURES as error:
            raise LLMError(classify_sdk_error(error, self.provider)) from error

    @override
    async def aclose(self) -> None:
        await self._sdk.close()

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        system_prompt = "\n\n".join(
            message.content for message in request.messages if message.role is Role.SYSTEM
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                message.model_dump(mode="json")
                for message in request.messages
                if message.role is not Role.SYSTEM
            ],
        }
        if system_prompt:
            payload["system"] = system_prompt
        if self.config.effort is not None:
            payload["output_config"] = {"effort": self.config.effort}
        return payload
