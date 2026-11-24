from pathlib import Path
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_client.settings import PROJECT_ROOT

MULTILINGUAL_MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", env_prefix="RAG_", extra="ignore"
    )

    data_dir: Path = PROJECT_ROOT / "data"
    vectorstore_dir: Path = PROJECT_ROOT / "vectorstore"
    embedding_cache_dir: Path = PROJECT_ROOT / ".cache" / "fastembed"
    collection_name: str = Field(default="nimbus_docs", min_length=3, max_length=63)
    embedding_model: str = MULTILINGUAL_MINILM
    chunk_size: int = Field(
        default=120, ge=20, le=512, description="tokens del modelo de embeddings por fragmento"
    )
    chunk_overlap: int = Field(default=20, ge=0, description="tokens compartidos entre fragmentos")
    top_k: int = Field(default=4, ge=3, le=5, description="fragmentos que se pasan al LLM")
    min_relevance: float = Field(
        default=0.45, ge=0.0, le=1.0, description="similitud coseno mínima para usar un fragmento"
    )

    @field_validator("data_dir", "vectorstore_dir", "embedding_cache_dir")
    @classmethod
    def relative_to_the_project(cls, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    @model_validator(mode="after")
    def overlap_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP tiene que ser menor que RAG_CHUNK_SIZE")
        return self
