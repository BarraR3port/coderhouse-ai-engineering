import argparse
import asyncio
import logging
import sys

from pydantic import ValidationError

from llm_client import (
    AsyncLLMManager,
    ChatMessage,
    ConfigurationError,
    LLMError,
    ModelResponse,
    Provider,
    Role,
    Settings,
)

QUESTION = "¿Qué es la entropía?"
CONVERSATION = [
    ChatMessage(role=Role.SYSTEM, content="Respondé en español, en un párrafo breve y claro."),
    ChatMessage(role=Role.USER, content=QUESTION),
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Pregunta '{QUESTION}' en modo normal y en streaming."
    )
    parser.add_argument(
        "--provider",
        type=Provider,
        choices=list(Provider),
        help="proveedor a usar; por defecto, todos los que tengan API key en el .env",
    )
    return parser.parse_args(argv)


def print_response(response: ModelResponse) -> None:
    if response.ok:
        print(response.content)
        tokens = (
            f"{response.usage.input_tokens} in / {response.usage.output_tokens} out"
            if response.usage
            else "sin datos de uso"
        )
        print(
            f"\n[{response.model} · {response.latency_ms:.0f} ms · {tokens} · "
            f"intentos: {response.attempts}]"
        )
    else:
        print("La llamada falló, pero el programa sigue. Error estructurado:")
        print(response.error.model_dump_json(indent=2))


async def ask(manager: AsyncLLMManager, provider: Provider) -> None:
    print(f"\n=== {provider} · modo normal ===")
    print_response(await manager.generate(CONVERSATION, provider=provider))

    print(f"\n=== {provider} · modo streaming ===")
    try:
        async for token in manager.stream(CONVERSATION, provider=provider):
            print(token, end="", flush=True)
        print()
    except LLMError as error:
        print("\nEl stream falló, pero el programa sigue. Error estructurado:")
        print(error.info.model_dump_json(indent=2))


async def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as error:
        print(f"Configuración inválida en el .env:\n{error}")
        return 2

    providers = [args.provider] if args.provider else settings.configured_providers()
    if not providers:
        print(
            "No hay API keys configuradas. Copiá .env.example a .env y completá "
            "OPENAI_API_KEY y/o ANTHROPIC_API_KEY."
        )
        return 1

    async with AsyncLLMManager(settings) as manager:
        for provider in providers:
            try:
                await ask(manager, provider)
            except ConfigurationError as error:
                print(f"\n=== {provider} ===\n{error}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    sys.exit(asyncio.run(main(sys.argv[1:])))
