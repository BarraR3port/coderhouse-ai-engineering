import copy
from abc import abstractmethod
from functools import cached_property
from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from tokenizers import Tokenizer

from rag.settings import RagSettings


class TokenAwareEmbeddings(Embeddings):
    @property
    @abstractmethod
    def max_tokens(self) -> int: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...


class LocalEmbeddings(TokenAwareEmbeddings):
    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir

    @cached_property
    def _model(self) -> TextEmbedding:
        return TextEmbedding(model_name=self.model_name, cache_dir=str(self.cache_dir))

    @cached_property
    def _tokenizer(self) -> Tokenizer:
        tokenizer: Tokenizer = self._model.model.tokenizer
        return tokenizer

    @cached_property
    def _counting_tokenizer(self) -> Tokenizer:
        counting = copy.deepcopy(self._tokenizer)
        counting.no_truncation()
        counting.no_padding()
        return counting

    @property
    def max_tokens(self) -> int:
        return int(self._tokenizer.truncation["max_length"])

    def count_tokens(self, text: str) -> int:
        return len(self._counting_tokenizer.encode(text).ids)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def local_embeddings(settings: RagSettings) -> LocalEmbeddings:
    return LocalEmbeddings(settings.embedding_model, settings.embedding_cache_dir)
