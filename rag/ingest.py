import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag.embeddings import TokenAwareEmbeddings
from rag.settings import RagSettings

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".md", ".txt")
MARKDOWN_HEADERS = [("#", "titulo"), ("##", "seccion")]
JUNK_CHARS = re.compile(r"[■□▪\x0c​]")
REPEATED_SPACES = re.compile(r"[ \t]+")
SPACES_AROUND_NEWLINE = re.compile(r"[ \t]*\n[ \t]*")
EXTRA_BLANK_LINES = re.compile(r"\n{3,}")

LengthFunction = Callable[[str], int]


class IngestError(RuntimeError):
    pass


class EmbeddingMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestReport:
    documents: int
    chunks: int
    reindexed: bool


def ingest(
    settings: RagSettings, embeddings: TokenAwareEmbeddings, *, rebuild: bool = False
) -> IngestReport:
    documents = load_documents(settings.data_dir)
    fingerprint = corpus_fingerprint(documents, settings)
    client = chromadb.PersistentClient(path=str(settings.vectorstore_dir))
    existing = find_collection(client, settings.collection_name)
    if existing is not None and not rebuild and is_up_to_date(existing, fingerprint):
        logger.info(
            "La colección '%s' ya está al día (%d fragmentos): no se reindexa",
            settings.collection_name,
            existing.count(),
        )
        return IngestReport(len(documents), existing.count(), reindexed=False)

    try:
        max_tokens = embeddings.max_tokens
    except Exception as error:
        raise IngestError(
            f"No se pudo cargar el modelo de embeddings '{settings.embedding_model}': {error}"
        ) from error
    if settings.chunk_size > max_tokens:
        raise IngestError(
            f"RAG_CHUNK_SIZE={settings.chunk_size} supera los {max_tokens} tokens que "
            f"lee '{settings.embedding_model}': el resto de cada fragmento no se vectorizaría"
        )
    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=embeddings.count_tokens,
    )
    if existing is not None:
        logger.info("Los documentos o la configuración cambiaron: se reconstruye la colección")
        client.delete_collection(settings.collection_name)
    store = Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
    )
    try:
        store.add_documents(chunks, ids=[chunk.id for chunk in chunks])
    except Exception as error:
        raise IngestError(f"No se pudieron vectorizar los fragmentos: {error}") from error
    collection = client.get_collection(settings.collection_name)
    # Chroma rechaza claves hnsw:* en modify(); la métrica ya quedó fija al crear la colección.
    metadata = {k: v for k, v in (collection.metadata or {}).items() if not k.startswith("hnsw:")}
    collection.modify(metadata={**metadata, "fingerprint": fingerprint})
    logger.info(
        "Indexados %d fragmentos de %d documentos en %s",
        len(chunks),
        len(documents),
        settings.vectorstore_dir,
    )
    return IngestReport(len(documents), len(chunks), reindexed=True)


def is_up_to_date(collection: chromadb.Collection, fingerprint: str) -> bool:
    return (collection.metadata or {}).get("fingerprint") == fingerprint and collection.count() > 0


def open_vectorstore(settings: RagSettings, embeddings: Embeddings) -> Chroma:
    client = chromadb.PersistentClient(path=str(settings.vectorstore_dir))
    collection = find_collection(client, settings.collection_name)
    if collection is None or collection.count() == 0:
        raise IngestError(
            f"La colección '{settings.collection_name}' está vacía: "
            "corré `python ingest.py` primero"
        )
    indexed_with = (collection.metadata or {}).get("embedding_model")
    if indexed_with != settings.embedding_model:
        raise EmbeddingMismatchError(
            f"La colección se indexó con '{indexed_with}' y se quiere consultar con "
            f"'{settings.embedding_model}': reindexá con `python ingest.py --rebuild`"
        )
    return Chroma(
        client=client, collection_name=settings.collection_name, embedding_function=embeddings
    )


def load_documents(data_dir: Path) -> list[Document]:
    if not data_dir.is_dir():
        raise IngestError(f"No existe la carpeta de documentos {data_dir}")
    paths = sorted(path for path in data_dir.iterdir() if path.suffix in SUPPORTED_SUFFIXES)
    if not paths:
        raise IngestError(f"{data_dir} no tiene archivos {' ni '.join(SUPPORTED_SUFFIXES)}")
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"archivo": path.name})
        for path in paths
    ]


def clean_text(text: str) -> str:
    text = JUNK_CHARS.sub("", text)
    text = REPEATED_SPACES.sub(" ", text)
    text = SPACES_AROUND_NEWLINE.sub("\n", text)
    text = EXTRA_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    length_function: LengthFunction,
) -> list[Document]:
    by_section = MarkdownHeaderTextSplitter(MARKDOWN_HEADERS, strip_headers=False)
    by_size = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
    )
    chunks: list[Document] = []
    for document in documents:
        archivo = document.metadata["archivo"]
        text = clean_text(document.page_content)
        sections = (
            by_section.split_text(text)
            if archivo.endswith(".md")
            else [Document(page_content=text)]
        )
        for section in sections:
            section.page_content = clean_text(section.page_content)
            section.metadata = {**section.metadata, "archivo": archivo}
        for index, chunk in enumerate(by_size.split_documents(sections)):
            chunk.id = f"{archivo}#{index:03d}"
            chunk.metadata["fragmento"] = index
            chunks.append(chunk)
    return chunks


def corpus_fingerprint(documents: list[Document], settings: RagSettings) -> str:
    payload = {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "documents": [
            [document.metadata["archivo"], sha256(document.page_content)] for document in documents
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_collection(client: ClientAPI, name: str) -> chromadb.Collection | None:
    try:
        return client.get_collection(name)
    except NotFoundError:
        return None
