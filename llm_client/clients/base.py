from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from time import perf_counter

from llm_client.schemas import ChatRequest, LLMConfig, ModelResponse, Provider


class BaseLLMClient(ABC):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @property
    def provider(self) -> Provider:
        return self.config.provider

    @abstractmethod
    async def generate(self, request: ChatRequest) -> ModelResponse: ...

    @abstractmethod
    def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]: ...

    @abstractmethod
    async def aclose(self) -> None: ...


def elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000
