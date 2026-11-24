import json
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field

from rag import RagSettings, TokenAwareEmbeddings

WORD = re.compile(r"\w+")
DIMENSIONS = 512

CORPUS = {
    "despliegues.md": """# Despliegues

## Rollback

El rollback es automático si la tasa de errores 5xx del canario supera el 1 %.

## Feature flags

Un feature flag no puede quedar activo más de 90 días.
""",
    "guardias.txt": "La guardia rota los lunes y se compensa con un día libre por semana.",
}


class KeywordEmbeddings(TokenAwareEmbeddings):
    @property
    def max_tokens(self) -> int:
        return 512

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in WORD.findall(text.lower()):
            if len(word) > 3:
                vector[sum(map(ord, word)) % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class AsyncOnlyChatModel(GenericFakeChatModel):
    received: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("la cadena RAG llamó al modelo de forma sincrónica")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(messages)
        return super()._generate(messages, stop=stop, **kwargs)


def chat_model(*replies: str) -> AsyncOnlyChatModel:
    messages: Iterator[AIMessage] = iter(AIMessage(content=reply) for reply in replies)
    return AsyncOnlyChatModel(messages=messages, received=[])


def model_reply(respuesta: str, fragmentos_usados: list[int]) -> str:
    return json.dumps(
        {"respuesta": respuesta, "fragmentos_usados": fragmentos_usados}, ensure_ascii=False
    )


def write_corpus(data_dir: Path, corpus: dict[str, str] = CORPUS) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, content in corpus.items():
        (data_dir / name).write_text(content, encoding="utf-8")
    return data_dir


def rag_settings(tmp_path: Path, **overrides: Any) -> RagSettings:
    values: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "vectorstore_dir": tmp_path / "vectorstore",
        "collection_name": "test_docs",
        "embedding_model": "keyword-embeddings",
        "chunk_size": 60,
        "chunk_overlap": 10,
        "top_k": 3,
        "min_relevance": 0.3,
        **overrides,
    }
    return RagSettings(_env_file=None, **values)
