import json
from functools import partial

import pytest

import run_pipeline as cli
from entity_pipeline import build_extraction_chain
from llm_client import Provider, Settings
from tests.fake_api import FakeAPI
from tests.http_mocks import openai_completion

VALID = {
    "tecnologias": ["PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "La base rechaza conexiones y el checkout devuelve 503.",
}


@pytest.fixture
def cli_against(
    monkeypatch: pytest.MonkeyPatch, fast_retry_settings: Settings, fake_api: FakeAPI
) -> FakeAPI:
    monkeypatch.setattr(cli, "Settings", lambda: fast_retry_settings)
    monkeypatch.setattr(
        cli,
        "build_extraction_chain",
        partial(build_extraction_chain, base_url=f"{fake_api.base_url}/v1"),
    )
    return fake_api


async def test_prints_the_validated_json_for_a_custom_text(
    cli_against: FakeAPI, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_against.respond((200, openai_completion(json.dumps(VALID))))

    exit_code = await cli.main(["--provider", "openai", "FATAL: too many clients already"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert json.loads(output.split("=== texto propio ===\n", 1)[1]) == VALID


async def test_keeps_going_with_a_structured_error_when_the_key_is_invalid(
    cli_against: FakeAPI, capsys: pytest.CaptureFixture[str]
) -> None:
    error = {"error": {"message": "bad key", "type": "invalid_request_error", "code": "x"}}
    cli_against.respond((401, error))

    exit_code = await cli.main(["--provider", "openai", "un log"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "La extracción falló, pero el programa sigue" in output
    assert '"error": "authentication"' in output
    assert '"status_code": 401' in output


async def test_stress_test_reports_each_invalid_sample_and_keeps_going(
    cli_against: FakeAPI, fast_retry_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    no_technologies = VALID | {"tecnologias": []}
    attempts = fast_retry_settings.llm_max_attempts * len(cli.SAMPLES)
    cli_against.respond(*[(200, openai_completion(json.dumps(no_technologies)))] * attempts)

    exit_code = await cli.main(["--provider", "openai"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert len(cli_against.requests) == attempts
    assert output.count('"error": "salida_invalida"') == len(cli.SAMPLES)
    assert output.count(
        '"detalle": "tecnologias: Value error, el texto tiene que nombrar al menos una tecnología"'
    ) == len(cli.SAMPLES)
    for name in cli.SAMPLES:
        assert f"=== {name} ===" in output


async def test_reports_a_refusal_and_keeps_going(
    cli_against: FakeAPI, capsys: pytest.CaptureFixture[str]
) -> None:
    body = openai_completion("")
    body["choices"][0]["message"] = {"role": "assistant", "content": None, "refusal": "No."}
    cli_against.respond((200, body))

    exit_code = await cli.main(["--provider", "openai", "un log"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"error": "sin_respuesta"' in output
    assert '"detalle": "el modelo se negó a responder: No."' in output


async def test_an_unexpected_error_in_one_sample_does_not_stop_the_others(
    cli_against: FakeAPI, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = openai_completion("")
    empty["choices"][0]["message"] = {"role": "assistant", "content": None}
    cli_against.respond((200, empty), *[(200, openai_completion(json.dumps(VALID)))] * 2)

    exit_code = await cli.main(["--provider", "openai"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.count('"error": "inesperado"') == 1
    assert output.count('"nivel_de_criticidad": "alta"') == 2


async def test_processes_every_sample_concurrently(
    cli_against: FakeAPI, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_against.respond(*[(200, openai_completion(json.dumps(VALID)))] * len(cli.SAMPLES))

    exit_code = await cli.main(["--provider", "openai"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert len(cli_against.requests) == len(cli.SAMPLES)
    for name in cli.SAMPLES:
        assert f"=== {name} ===" in output


async def test_rejects_a_blank_custom_text(
    cli_against: FakeAPI, capsys: pytest.CaptureFixture[str]
) -> None:
    assert await cli.main(["--provider", "openai", "   "]) == 2
    assert cli_against.requests == []


async def test_exits_with_a_hint_when_no_key_is_configured(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None))

    assert await cli.main([]) == 1
    assert ".env.example" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("keys", "configured_default", "expected"),
    [
        ({"openai_api_key": "sk", "anthropic_api_key": "sk"}, Provider.ANTHROPIC, "anthropic"),
        ({"anthropic_api_key": "sk"}, Provider.OPENAI, "anthropic"),
        ({}, Provider.OPENAI, None),
    ],
)
def test_choose_provider_prefers_the_configured_default(
    keys: dict[str, str], configured_default: Provider, expected: str | None
) -> None:
    settings = Settings(_env_file=None, llm_provider=configured_default, **keys)

    assert cli.choose_provider(settings, None) == expected
