from collections.abc import AsyncGenerator
from time import perf_counter
from typing import Any, override

import httpx2
from openai import AsyncOpenAI

from llm_client.clients.base import BaseLLMClient, elapsed_ms
from llm_client.errors import PROVIDER_FAILURES, LLMError, classify_sdk_error
from llm_client.schemas import ChatRequest, LLMConfig, ModelResponse, Usage


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: LLMConfig, http_client: httpx2.AsyncClient | None = None) -> None:
        super().__init__(config)
        self._sdk = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            timeout=config.timeout_seconds,
            max_retries=0,
            http_client=http_client,
        )

    @override
    async def generate(self, request: ChatRequest) -> ModelResponse:
        started = perf_counter()
        try:
            completion = await self._sdk.chat.completions.create(**self._payload(request))
        except PROVIDER_FAILURES as error:
            raise LLMError(classify_sdk_error(error, self.provider)) from error

        choice = completion.choices[0]
        usage = completion.usage
        return ModelResponse(
            provider=self.provider,
            model=completion.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=Usage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens)
            if usage
            else None,
            latency_ms=elapsed_ms(started),
        )

    @override
    async def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        try:
            chunks = await self._sdk.chat.completions.create(**self._payload(request), stream=True)
            async with chunks:
                async for chunk in chunks:
                    if chunk.choices and (token := chunk.choices[0].delta.content):
                        yield token
        except PROVIDER_FAILURES as error:
            raise LLMError(classify_sdk_error(error, self.provider)) from error

    @override
    async def aclose(self) -> None:
        await self._sdk.close()

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        return payload
