import itertools
from pathlib import Path

import chromadb
import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from llm_client.settings import PROJECT_ROOT
from rag import EmbeddingMismatchError, IngestError, RagSettings, ingest, open_vectorstore
from rag.ingest import clean_text, load_documents, split_documents
from tests.rag_fakes import KeywordEmbeddings, rag_settings, write_corpus

word_count = KeywordEmbeddings().count_tokens


def ingest_corpus(settings: RagSettings, **kwargs: object):
    return ingest(settings, KeywordEmbeddings(), **kwargs)


def test_markdown_is_split_by_section_with_the_section_in_the_metadata(tmp_path: Path) -> None:
    chunks = split_documents(
        load_documents(write_corpus(tmp_path / "data")),
        chunk_size=60,
        chunk_overlap=10,
        length_function=word_count,
    )

    secciones = [(chunk.metadata["archivo"], chunk.metadata.get("seccion")) for chunk in chunks]
    assert secciones == [
        ("despliegues.md", "Rollback"),
        ("despliegues.md", "Feature flags"),
        ("guardias.txt", None),
    ]
    assert "## Rollback\nEl rollback es automático" in chunks[0].page_content
    assert "Feature flags" not in chunks[0].page_content


def test_chunk_ids_are_deterministic(tmp_path: Path) -> None:
    documents = load_documents(write_corpus(tmp_path / "data"))

    def ids() -> list[str | None]:
        chunks = split_documents(
            documents, chunk_size=60, chunk_overlap=10, length_function=word_count
        )
        return [chunk.id for chunk in chunks]

    assert ids() == ids() == ["despliegues.md#000", "despliegues.md#001", "guardias.txt#000"]


def test_long_sections_are_cut_by_length_with_overlap() -> None:
    sentences = [f"Oración número {n} sobre la política de guardias." for n in range(1, 13)]
    document = Document(page_content=" ".join(sentences), metadata={"archivo": "largo.txt"})

    chunks = split_documents([document], chunk_size=30, chunk_overlap=8, length_function=word_count)

    assert len(chunks) > 1
    assert all(word_count(chunk.page_content) <= 30 for chunk in chunks)
    for previous, current in itertools.pairwise(chunks):
        assert previous.page_content.split()[-1] in current.page_content.split()[:8]


def test_clean_text_removes_junk_and_extra_blank_lines() -> None:
    assert (
        clean_text("Título■■\n\n\n\nTexto   con    espacios  \n") == "Título\n\nTexto con espacios"
    )


def test_ingest_stores_every_chunk_with_its_source(tmp_path: Path) -> None:
    settings = rag_settings(tmp_path)
    write_corpus(settings.data_dir)

    report = ingest_corpus(settings)

    assert (report.documents, report.chunks, report.reindexed) == (2, 3, True)
    stored = open_vectorstore(settings, KeywordEmbeddings()).get()
    assert sorted(stored["ids"]) == ["despliegues.md#000", "despliegues.md#001", "guardias.txt#000"]
    assert {metadata["archivo"] for metadata in stored["metadatas"]} == {
        "despliegues.md",
        "guardias.txt",
    }


def test_a_second_ingest_reuses_the_persisted_collection(tmp_path: Path) -> None:
    settings = rag_settings(tmp_path)
    write_corpus(settings.data_dir)
    ingest_corpus(settings)

    report = ingest_corpus(settings)

    assert (report.chunks, report.reindexed) == (3, False)


def test_ingest_rebuilds_when_a_document_changes(tmp_path: Path) -> None:
    settings = rag_settings(tmp_path)
    write_corpus(settings.data_dir)
    ingest_corpus(settings)
    (settings.data_dir / "guardias.txt").write_text("La guardia ahora rota los martes.")

    report = ingest_corpus(settings)

    assert report.reindexed
    stored = open_vectorstore(settings, KeywordEmbeddings()).get(ids=["guardias.txt#000"])
    assert stored["documents"] == ["La guardia ahora rota los martes."]


def test_rebuild_reindexes_even_without_changes(tmp_path: Path) -> None:
    settings = rag_settings(tmp_path)
    write_corpus(settings.data_dir)
    ingest_corpus(settings)

    report = ingest_corpus(settings, rebuild=True)

    assert (report.chunks, report.reindexed) == (3, True)


def test_querying_with_another_embedding_model_fails(tmp_path: Path) -> None:
    settings = rag_settings(tmp_path)
    write_corpus(settings.data_dir)
    ingest_corpus(settings)
    other_model = settings.model_copy(update={"embedding_model": "otro-modelo"})

    with pytest.raises(EmbeddingMismatchError, match="keyword-embeddings"):
        open_vectorstore(other_model, KeywordEmbeddings())


def test_querying_before_ingesting_explains_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match=r"ingest\.py"):
        open_vectorstore(rag_settings(tmp_path), KeywordEmbeddings())


def test_an_empty_data_folder_is_reported(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()

    with pytest.raises(IngestError, match="no tiene archivos"):
        load_documents(tmp_path / "data")


@pytest.mark.parametrize(
    "overrides",
    [{"top_k": 2}, {"top_k": 6}, {"chunk_size": 100, "chunk_overlap": 100}],
)
def test_settings_reject_values_outside_the_allowed_range(
    tmp_path: Path, overrides: dict[str, int]
) -> None:
    with pytest.raises(ValidationError):
        rag_settings(tmp_path, **overrides)


def test_relative_folders_are_resolved_from_the_project_root(tmp_path: Path) -> None:
    settings = RagSettings(_env_file=None, data_dir="data", vectorstore_dir="vectorstore")

    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.vectorstore_dir == PROJECT_ROOT / "vectorstore"


class BrokenEmbeddings(KeywordEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise ValueError("Could not load model from any source")


def test_a_failed_ingest_leaves_nothing_marked_as_up_to_date(tmp_path: Path) -> None:
    settings = rag_settings(tmp_path)
    write_corpus(settings.data_dir)

    with pytest.raises(IngestError, match="No se pudieron vectorizar"):
        ingest(settings, BrokenEmbeddings())
    report = ingest_corpus(settings)

    assert (report.chunks, report.reindexed) == (3, True)


def test_an_empty_collection_is_never_up_to_date(tmp_path: Path) -> None:
    settings = rag_settings(tmp_path)
    write_corpus(settings.data_dir)
    ingest_corpus(settings)
    collection = chromadb.PersistentClient(path=str(settings.vectorstore_dir)).get_collection(
        settings.collection_name
    )
    collection.delete(ids=collection.get()["ids"])

    with pytest.raises(IngestError, match="vacía"):
        open_vectorstore(settings, KeywordEmbeddings())
    assert ingest_corpus(settings).reindexed


class ShortContextEmbeddings(KeywordEmbeddings):
    @property
    def max_tokens(self) -> int:
        return 40


def test_chunks_cannot_exceed_what_the_embedding_model_reads(tmp_path: Path) -> None:
    settings = rag_settings(tmp_path, chunk_size=60)
    write_corpus(settings.data_dir)

    with pytest.raises(IngestError, match="40 tokens"):
        ingest(settings, ShortContextEmbeddings())
