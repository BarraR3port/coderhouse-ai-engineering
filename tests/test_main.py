import httpx2
import pytest

import main as cli
from llm_client import AnthropicClient, AsyncLLMManager, Settings
from tests.http_mocks import (
    RecordingTransport,
    anthropic_message,
    http_client,
    interrupted_sse,
    json_response,
)


async def no_wait(_: float) -> None:
    return None


@pytest.fixture
def isolated_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None))


async def test_main_keeps_going_when_the_network_fails_during_streaming(
    isolated_cli: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    transport = RecordingTransport(
        [json_response(200, anthropic_message("Es una medida del desorden."))]
        + [interrupted_sse([], httpx2.ReadTimeout("lento")) for _ in range(3)]
    )
    monkeypatch.setattr(
        cli,
        "AsyncLLMManager",
        lambda settings: AsyncLLMManager(
            settings,
            client_builder=lambda config: AnthropicClient(
                config, http_client=http_client(transport)
            ),
            sleep=no_wait,
        ),
    )

    exit_code = await cli.main(["--provider", "anthropic"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Es una medida del desorden." in output
    assert "El stream falló, pero el programa sigue" in output
    assert '"kind": "timeout"' in output
    assert '"attempts": 3' in output


async def test_main_explains_how_to_configure_keys_when_there_are_none(
    isolated_cli: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = await cli.main([])

    assert exit_code == 1
    assert ".env.example" in capsys.readouterr().out


async def test_main_reports_an_invalid_env_value_without_a_traceback(
    isolated_cli: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_DELAY_SECONDS", "30")

    exit_code = await cli.main([])

    assert exit_code == 2
    assert "Configuración inválida" in capsys.readouterr().out
