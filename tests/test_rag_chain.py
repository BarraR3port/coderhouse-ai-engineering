import logging
from pathlib import Path
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from entity_pipeline import build_chat_model
from llm_client import Provider, RetryPolicy, Settings
from rag import NO_LO_SE, RagChain, RagSettings, build_rag_chain, get_rag_response, open_vectorstore
from rag.ingest import ingest
from tests.fake_api import FakeAPI
from tests.http_mocks import openai_completion
from tests.rag_fakes import (
    KeywordEmbeddings,
    chat_model,
    model_reply,
    rag_settings,
    write_corpus,
)

ROLLBACK_QUESTION = "¿Cuándo es automático el rollback del canario?"
ROLLBACK_ANSWER = "Cuando la tasa de errores 5xx del canario supera el 1 %."


@pytest.fixture
def indexed(tmp_path: Path) -> RagSettings:
    settings = rag_settings(tmp_path)
    write_corpus(settings.data_dir)
    ingest(settings, KeywordEmbeddings())
    return settings


FAST_RETRY = RetryPolicy(max_attempts=3, base_delay_seconds=0.01, max_delay_seconds=0.02)


def chain_for(settings: RagSettings, llm: BaseChatModel) -> RagChain:
    return build_rag_chain(
        open_vectorstore(settings, KeywordEmbeddings()),
        llm,
        provider=Provider.OPENAI,
        retry=FAST_RETRY,
        top_k=settings.top_k,
        min_relevance=settings.min_relevance,
    )


async def test_the_answer_cites_its_sources_from_the_chunk_metadata(indexed: RagSettings) -> None:
    model = chat_model(model_reply(ROLLBACK_ANSWER, [1]))

    respuesta = await get_rag_response(chain_for(indexed, model), ROLLBACK_QUESTION)

    assert respuesta.encontrada
    assert respuesta.respuesta == ROLLBACK_ANSWER
    [fuente] = respuesta.fuentes
    assert (fuente.archivo, fuente.seccion, fuente.fragmento_id) == (
        "despliegues.md",
        "Rollback",
        "despliegues.md#000",
    )
    assert 0.3 <= fuente.relevancia <= 1.0


async def test_the_model_gets_the_question_and_only_the_relevant_context(
    indexed: RagSettings,
) -> None:
    model = chat_model(model_reply(ROLLBACK_ANSWER, [1]))

    await get_rag_response(chain_for(indexed, model), ROLLBACK_QUESTION)

    [[system, human]] = model.received
    assert NO_LO_SE in system.content
    assert "fragmentos_usados" in system.content
    assert "[1] (despliegues.md > Rollback)" in human.content
    assert "errores 5xx del canario supera el 1 %" in human.content
    assert "guardia" not in human.content
    assert human.content.endswith(f"PREGUNTA: {ROLLBACK_QUESTION}")


async def test_a_question_outside_the_documents_gets_no_lo_se_without_calling_the_model(
    indexed: RagSettings,
) -> None:
    model = chat_model()

    respuesta = await get_rag_response(
        chain_for(indexed, model), "¿Cuál es la capital de Australia?"
    )

    assert (respuesta.respuesta, respuesta.encontrada, respuesta.fuentes) == (NO_LO_SE, False, [])
    assert model.received == []


async def test_when_the_model_says_no_lo_se_there_are_no_sources(indexed: RagSettings) -> None:
    model = chat_model(model_reply("No lo sé.", []))

    respuesta = await get_rag_response(
        chain_for(indexed, model), "¿Quién aprueba el rollback del canario?"
    )

    assert (respuesta.respuesta, respuesta.encontrada, respuesta.fuentes) == (NO_LO_SE, False, [])


@pytest.mark.parametrize(
    "bad_reply",
    [
        model_reply(ROLLBACK_ANSWER, [7]),
        model_reply(ROLLBACK_ANSWER, []),
        "El rollback es automático.",
    ],
    ids=["cita-un-fragmento-que-no-existe", "responde-sin-citar", "no-es-json"],
)
async def test_an_invalid_model_output_is_retried(indexed: RagSettings, bad_reply: str) -> None:
    model = chat_model(bad_reply, model_reply(ROLLBACK_ANSWER, [1]))

    respuesta = await get_rag_response(chain_for(indexed, model), ROLLBACK_QUESTION)

    assert respuesta.fuentes[0].fragmento_id == "despliegues.md#000"
    assert len(model.received) == 2


async def test_the_error_surfaces_when_every_attempt_is_invalid(indexed: RagSettings) -> None:
    model = chat_model("no es json", "tampoco", "nada")

    with pytest.raises(OutputParserException):
        await get_rag_response(chain_for(indexed, model), ROLLBACK_QUESTION)
    assert len(model.received) == 3


async def test_an_empty_question_is_rejected_before_searching(indexed: RagSettings) -> None:
    model = chat_model()

    with pytest.raises(ValidationError):
        await get_rag_response(chain_for(indexed, model), "   ")
    assert model.received == []


def openai_chain(settings: RagSettings, fake_api: FakeAPI, llm_settings: Settings) -> RagChain:
    llm = build_chat_model(
        llm_settings.config_for(Provider.OPENAI), base_url=f"{fake_api.base_url}/v1"
    )
    return chain_for(settings, llm)


async def test_the_chain_runs_against_the_openai_api_without_leaking_the_key(
    indexed: RagSettings, fake_api: FakeAPI, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    fake_api.respond((200, openai_completion(model_reply(ROLLBACK_ANSWER, [1]))))

    respuesta = await get_rag_response(openai_chain(indexed, fake_api, settings), ROLLBACK_QUESTION)

    assert respuesta.fuentes[0].archivo == "despliegues.md"
    [request] = fake_api.requests
    assert request.headers["authorization"] == "Bearer sk-test-openai"
    system, user = request.body["messages"]
    assert system["role"] == "system" and NO_LO_SE in system["content"]
    assert user["role"] == "user" and "despliegues.md > Rollback" in user["content"]
    assert "sk-test-openai" not in caplog.text
    assert "sk-test-openai" not in respuesta.model_dump_json()
    assert "sk-test-openai" not in repr(settings)


async def test_a_rate_limit_from_the_provider_is_retried(
    indexed: RagSettings, fake_api: FakeAPI, settings: Settings
) -> None:
    rate_limited = {"error": {"message": "Rate limit reached", "type": "requests", "code": None}}
    fake_api.respond(
        (429, rate_limited), (200, openai_completion(model_reply(ROLLBACK_ANSWER, [1])))
    )

    respuesta = await get_rag_response(openai_chain(indexed, fake_api, settings), ROLLBACK_QUESTION)

    assert respuesta.encontrada
    assert len(fake_api.requests) == 2


async def test_retrieval_uses_the_async_vector_store_api(
    indexed: RagSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = open_vectorstore(indexed, KeywordEmbeddings())
    awaited: list[str] = []
    original = store.asimilarity_search_with_relevance_scores

    async def spy(query: str, **kwargs: Any) -> Any:
        awaited.append(query)
        return await original(query, **kwargs)

    monkeypatch.setattr(store, "asimilarity_search_with_relevance_scores", spy)
    chain = build_rag_chain(
        store,
        chat_model(model_reply(ROLLBACK_ANSWER, [1])),
        provider=Provider.OPENAI,
        retry=FAST_RETRY,
        top_k=indexed.top_k,
        min_relevance=indexed.min_relevance,
    )

    await get_rag_response(chain, ROLLBACK_QUESTION)

    assert awaited == [ROLLBACK_QUESTION]
