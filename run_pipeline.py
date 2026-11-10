import argparse
import asyncio
import json
import logging
import sys

from pydantic import ValidationError

from entity_pipeline import (
    OUTPUT_ERRORS,
    REFUSAL_ERRORS,
    ExtractionChain,
    SolicitudDeExtraccion,
    build_extraction_chain,
    extract,
)
from entity_pipeline.chain import describe_output_error, first_line
from llm_client import ConfigurationError, Provider, Settings
from llm_client.errors import PROVIDER_FAILURES, classify_sdk_error

SAMPLES = {
    "arquitectura": (
        "Nuestra API de pedidos está hecha con FastAPI y publica cada orden en un tópico de Kafka. "
        "Un worker en Go consume esos eventos y guarda los totales en PostgreSQL; Redis cachea las "
        "consultas más pedidas y todo corre en Kubernetes."
    ),
    "log de error": (
        "2026-11-10T03:12:44Z ERROR payment-service psycopg.OperationalError: connection to server "
        'at "db-primary" (10.0.3.7), port 5432 failed: FATAL: sorry, too many clients already. '
        "Desde hace 15 minutos el checkout responde 503 y no se están cobrando los pedidos."
    ),
    "texto ambiguo": (
        "Anoche algo se puso lento, capaz fue la base o la red, nadie sabe bien. "
        "Hoy parece que anda."
    ),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae tecnologías, criticidad y un resumen técnico de textos de ejemplo."
    )
    parser.add_argument(
        "texto",
        nargs="?",
        help="texto propio a analizar; sin él, se procesan los ejemplos de SAMPLES",
    )
    parser.add_argument(
        "--provider",
        type=Provider,
        choices=list(Provider),
        help="proveedor a usar; por defecto, LLM_PROVIDER si tiene API key",
    )
    return parser.parse_args(argv)


def choose_provider(settings: Settings, requested: Provider | None) -> Provider | None:
    if requested is not None:
        return requested
    configured = settings.configured_providers()
    if settings.llm_provider in configured:
        return settings.llm_provider
    return configured[0] if configured else None


async def report(chain: ExtractionChain, provider: Provider, name: str, texto: str) -> str:
    try:
        extraction = await extract(chain, texto)
    except OUTPUT_ERRORS as error:
        failure = {"error": "salida_invalida", "detalle": describe_output_error(error)}
    except REFUSAL_ERRORS as error:
        failure = {"error": "sin_respuesta", "detalle": describe_output_error(error)}
    except PROVIDER_FAILURES as error:
        info = classify_sdk_error(error, provider)
        failure = {"error": info.kind, "detalle": info.message, "status_code": info.status_code}
    except Exception as error:
        failure = {"error": "inesperado", "detalle": f"{type(error).__name__}: {first_line(error)}"}
    else:
        return f"=== {name} ===\n{extraction.model_dump_json(indent=2)}"
    details = json.dumps(failure, indent=2, ensure_ascii=False)
    return f"=== {name} ===\nLa extracción falló, pero el programa sigue:\n{details}"


async def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as error:
        print(f"Configuración inválida en el .env:\n{error}")
        return 2

    provider = choose_provider(settings, args.provider)
    if provider is None:
        print(
            "No hay API keys configuradas. Copiá .env.example a .env y completá "
            "OPENAI_API_KEY y/o ANTHROPIC_API_KEY."
        )
        return 1
    try:
        chain = build_extraction_chain(settings, provider)
    except ConfigurationError as error:
        print(error)
        return 1

    if args.texto is not None:
        try:
            samples = {"texto propio": SolicitudDeExtraccion(texto=args.texto).texto}
        except ValidationError as error:
            print(f"El texto no es válido:\n{error}")
            return 2
    else:
        samples = SAMPLES
    reports = await asyncio.gather(
        *(report(chain, provider, name, texto) for name, texto in samples.items())
    )
    print(f"\nProveedor: {provider}\n")
    print("\n\n".join(reports))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.exit(asyncio.run(main(sys.argv[1:])))
