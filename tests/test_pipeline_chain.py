import json
import logging

import anthropic
import openai
import pytest
from langchain_core.exceptions import ModelRefusalError
from pydantic import ValidationError

from entity_pipeline import (
    Criticidad,
    ExtraccionTecnica,
    IncompleteOutputError,
    build_extraction_chain,
    extract,
)
from entity_pipeline.chain import format_instructions
from llm_client import Provider, Settings
from tests.fake_api import FakeAPI, ScriptedResponse
from tests.http_mocks import anthropic_error_body, anthropic_message, openai_completion

VALID = {
    "tecnologias": ["FastAPI", "Kafka"],
    "nivel_de_criticidad": "media",
    "resumen_tecnico": "La API publica pedidos en Kafka.",
}
NO_TECHNOLOGIES = VALID | {"tecnologias": []}
SCHEMA_DETAIL = "tecnologias: Value error, el texto tiene que nombrar al menos una tecnología"
TEXT = "Nuestra API en FastAPI publica cada pedido en Kafka."


def openai_reply(payload: dict | str = VALID, finish_reason: str = "stop") -> ScriptedResponse:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    body = openai_completion(content)
    body["choices"][0]["finish_reason"] = finish_reason
    return 200, body


def openai_refusal() -> ScriptedResponse:
    status, body = openai_reply()
    body["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "refusal": "No puedo ayudar con eso.",
    }
    return status, body


def openai_failure(status: int, code: str) -> ScriptedResponse:
    return status, {"error": {"message": f"simulated {code}", "type": code, "code": code}}


def anthropic_reply(payload: dict | str = VALID, stop_reason: str = "end_turn") -> ScriptedResponse:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return 200, anthropic_message(content) | {"stop_reason": stop_reason}


async def run_openai(api: FakeAPI, settings: Settings) -> ExtraccionTecnica:
    chain = build_extraction_chain(settings, Provider.OPENAI, base_url=f"{api.base_url}/v1")
    return await extract(chain, TEXT)


async def run_anthropic(api: FakeAPI, settings: Settings) -> ExtraccionTecnica:
    chain = build_extraction_chain(settings, Provider.ANTHROPIC, base_url=api.base_url)
    return await extract(chain, TEXT)


async def test_openai_returns_a_validated_extraction(
    fake_api: FakeAPI, fast_retry_settings: Settings
) -> None:
    fake_api.respond(openai_reply())

    result = await run_openai(fake_api, fast_retry_settings)

    assert result == ExtraccionTecnica.model_validate(VALID)
    assert result.nivel_de_criticidad is Criticidad.MEDIA


async def test_openai_request_uses_strict_json_schema_and_the_prompt_template(
    fake_api: FakeAPI, fast_retry_settings: Settings
) -> None:
    fake_api.respond(openai_reply())

    await run_openai(fake_api, fast_retry_settings)

    sent = fake_api.sent()
    assert fake_api.requests[0].path == "/v1/chat/completions"
    assert sent["temperature"] == 0
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["name"] == "ExtraccionTecnica"
    assert sent["response_format"]["json_schema"]["strict"] is True
    system, user = sent["messages"]
    assert system["role"] == "system"
    assert format_instructions(ExtraccionTecnica) in system["content"]
    assert user == {"role": "user", "content": TEXT}


def test_format_instructions_describe_every_field_of_the_contract() -> None:
    instructions = format_instructions(ExtraccionTecnica)

    for name, field in ExtraccionTecnica.model_fields.items():
        assert f"- {name}: {field.description}" in instructions


@pytest.mark.parametrize(
    ("bad_reply", "logged"),
    [
        (openai_reply("esto no es JSON"), "La salida no cumple el esquema: Invalid JSON"),
        (openai_reply(NO_TECHNOLOGIES), f"La salida no cumple el esquema: {SCHEMA_DETAIL}"),
        (
            openai_reply('{"tecnologias": ["Fast', finish_reason="length"),
            "El modelo cortó la respuesta: el objeto quedó incompleto (terminó por length)",
        ),
    ],
    ids=["json-mal-formado", "viola-el-esquema", "cortado-por-tokens"],
)
async def test_openai_retries_invalid_or_incomplete_output(
    fake_api: FakeAPI,
    fast_retry_settings: Settings,
    bad_reply: ScriptedResponse,
    logged: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_api.respond(bad_reply, openai_reply())

    with caplog.at_level(logging.INFO, logger="entity_pipeline"):
        result = await run_openai(fake_api, fast_retry_settings)

    assert result.tecnologias == ["FastAPI", "Kafka"]
    assert len(fake_api.requests) == 2
    assert logged in caplog.text
    assert "Reintentando la extracción: intento 2" in caplog.text


async def test_openai_gives_up_after_the_configured_attempts(
    fake_api: FakeAPI, fast_retry_settings: Settings
) -> None:
    fake_api.respond(*[openai_reply(NO_TECHNOLOGIES)] * 3)

    with pytest.raises(ValidationError, match="al menos una tecnología"):
        await run_openai(fake_api, fast_retry_settings)

    assert len(fake_api.requests) == fast_retry_settings.llm_max_attempts


@pytest.mark.parametrize(
    "transient",
    [openai_failure(500, "server_error"), openai_failure(429, "rate_limit_exceeded")],
    ids=["500", "429-rate-limit"],
)
async def test_openai_retries_transient_provider_errors(
    fake_api: FakeAPI, fast_retry_settings: Settings, transient: ScriptedResponse
) -> None:
    fake_api.respond(transient, openai_reply())

    await run_openai(fake_api, fast_retry_settings)

    assert len(fake_api.requests) == 2


@pytest.mark.parametrize(
    ("permanent", "expected"),
    [
        (openai_failure(401, "invalid_api_key"), openai.AuthenticationError),
        (openai_failure(429, "insufficient_quota"), openai.RateLimitError),
    ],
    ids=["401", "429-sin-cuota"],
)
async def test_openai_does_not_retry_permanent_errors(
    fake_api: FakeAPI,
    fast_retry_settings: Settings,
    permanent: ScriptedResponse,
    expected: type[Exception],
) -> None:
    fake_api.respond(permanent)

    with pytest.raises(expected):
        await run_openai(fake_api, fast_retry_settings)

    assert len(fake_api.requests) == 1


async def test_anthropic_returns_a_validated_extraction_with_native_structured_output(
    fake_api: FakeAPI, fast_retry_settings: Settings
) -> None:
    fake_api.respond(anthropic_reply())

    result = await run_anthropic(fake_api, fast_retry_settings)

    assert result == ExtraccionTecnica.model_validate(VALID)
    sent = fake_api.sent()
    assert fake_api.requests[0].path == "/v1/messages"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["effort"] == "low"
    assert "tool_choice" not in sent
    assert "temperature" not in sent
    assert format_instructions(ExtraccionTecnica) in sent["system"]
    assert sent["messages"] == [{"role": "user", "content": TEXT}]


@pytest.mark.parametrize(
    ("bad_reply", "logged"),
    [
        (anthropic_reply("esto no es JSON"), "La salida no cumple el esquema: Invalid json output"),
        (anthropic_reply(NO_TECHNOLOGIES), f"La salida no cumple el esquema: {SCHEMA_DETAIL}"),
        (
            anthropic_reply('{"tecnologias": ["Fast', stop_reason="max_tokens"),
            "El modelo cortó la respuesta: el objeto quedó incompleto (terminó por max_tokens)",
        ),
        (
            anthropic_reply(VALID, stop_reason="model_context_window_exceeded"),
            "El modelo cortó la respuesta: el objeto quedó incompleto "
            "(terminó por model_context_window_exceeded)",
        ),
        (
            (529, anthropic_error_body("overloaded_error")),
            "Falló la llamada al modelo: AnthropicOverloadedError",
        ),
    ],
    ids=[
        "json-mal-formado",
        "viola-el-esquema",
        "cortado-por-tokens",
        "sin-contexto",
        "529-sobrecarga",
    ],
)
async def test_anthropic_retries_invalid_output_and_overload(
    fake_api: FakeAPI,
    fast_retry_settings: Settings,
    bad_reply: ScriptedResponse,
    logged: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_api.respond(bad_reply, anthropic_reply())

    with caplog.at_level(logging.INFO, logger="entity_pipeline"):
        result = await run_anthropic(fake_api, fast_retry_settings)

    assert result.tecnologias == ["FastAPI", "Kafka"]
    assert len(fake_api.requests) == 2
    assert logged in caplog.text


async def test_anthropic_reports_incomplete_output_when_every_attempt_is_cut(
    fake_api: FakeAPI, fast_retry_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    fake_api.respond(*[anthropic_reply('{"tecnologias": [', stop_reason="max_tokens")] * 3)

    with (
        caplog.at_level(logging.WARNING, logger="entity_pipeline"),
        pytest.raises(IncompleteOutputError, match="max_tokens"),
    ):
        await run_anthropic(fake_api, fast_retry_settings)

    assert caplog.text.count("El modelo cortó la respuesta: el objeto quedó incompleto") == 3


@pytest.mark.parametrize(
    ("permanent", "expected"),
    [
        ((401, anthropic_error_body("authentication_error")), anthropic.AuthenticationError),
        ((402, anthropic_error_body("billing_error")), anthropic.APIStatusError),
    ],
    ids=["401", "402-sin-credito"],
)
async def test_anthropic_does_not_retry_permanent_errors(
    fake_api: FakeAPI,
    fast_retry_settings: Settings,
    permanent: ScriptedResponse,
    expected: type[Exception],
) -> None:
    fake_api.respond(permanent)

    with pytest.raises(expected):
        await run_anthropic(fake_api, fast_retry_settings)

    assert len(fake_api.requests) == 1


@pytest.mark.parametrize(
    ("provider", "reply", "logged"),
    [
        (
            Provider.OPENAI,
            openai_refusal(),
            "el modelo se negó a responder: No puedo ayudar con eso.",
        ),
        (
            Provider.OPENAI,
            openai_reply(finish_reason="content_filter"),
            "el filtro de contenido del proveedor bloqueó la respuesta",
        ),
        (
            Provider.ANTHROPIC,
            anthropic_reply("", stop_reason="refusal"),
            "el modelo se negó a responder",
        ),
    ],
    ids=["openai-rechazo", "openai-filtro", "anthropic-rechazo"],
)
async def test_refusals_are_reported_without_retrying(
    fake_api: FakeAPI,
    fast_retry_settings: Settings,
    provider: Provider,
    reply: ScriptedResponse,
    logged: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_api.respond(reply)
    run = run_openai if provider is Provider.OPENAI else run_anthropic

    with (
        caplog.at_level(logging.WARNING, logger="entity_pipeline"),
        pytest.raises((ModelRefusalError, openai.ContentFilterFinishReasonError)),
    ):
        await run(fake_api, fast_retry_settings)

    assert len(fake_api.requests) == 1
    assert f"El modelo no respondió: {logged}" in caplog.text
    assert "no cumple el esquema" not in caplog.text


async def test_the_flow_is_logged_step_by_step(
    fake_api: FakeAPI, fast_retry_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    fake_api.respond(openai_reply())

    with caplog.at_level(logging.INFO, logger="entity_pipeline"):
        await run_openai(fake_api, fast_retry_settings)

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Llamando a gpt-4o-mini",
        "Respuesta recibida: fin=stop, tokens=12 entrada / 30 salida",
        "Salida validada: 2 tecnologías, criticidad media",
    ]


async def test_blank_text_is_rejected_before_calling_the_model(
    fake_api: FakeAPI, fast_retry_settings: Settings
) -> None:
    chain = build_extraction_chain(fast_retry_settings, Provider.OPENAI, base_url=fake_api.base_url)

    with pytest.raises(ValidationError, match="texto"):
        await extract(chain, "   ")

    assert fake_api.requests == []
