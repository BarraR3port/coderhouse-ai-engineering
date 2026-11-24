import argparse
import asyncio
import json
import logging
import sys

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from entity_pipeline import build_chat_model
from entity_pipeline.chain import first_line
from llm_client import ConfigurationError, Provider, Settings
from llm_client.errors import PROVIDER_FAILURES, classify_sdk_error
from rag import (
    EmbeddingMismatchError,
    IngestError,
    RagChain,
    RagSettings,
    build_rag_chain,
    get_rag_response,
    ingest,
    local_embeddings,
    open_vectorstore,
    relevant,
    search,
)

SAMPLE_QUESTIONS = (
    "¿Qué pasa si durante el canario la tasa de errores 5xx supera el 1 %?",
    "¿Cuánto tiempo tengo para confirmar una alerta de un SEV2?",
    "¿Cuál es la capital de Australia?",
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Responde preguntas sobre la documentación de data/ con RAG."
    )
    parser.add_argument(
        "pregunta", nargs="?", help="pregunta propia; sin ella, se usan las de ejemplo"
    )
    parser.add_argument(
        "--provider",
        type=Provider,
        choices=list(Provider),
        help="proveedor del LLM; por defecto, LLM_PROVIDER si tiene API key",
    )
    parser.add_argument(
        "--solo-recuperar",
        action="store_true",
        help="muestra los fragmentos que se recuperarían, sin llamar al LLM (no requiere API key)",
    )
    return parser.parse_args(argv)


async def answer(chain: RagChain, provider: Provider, pregunta: str) -> str:
    try:
        respuesta = await get_rag_response(chain, pregunta)
    except ValidationError as error:
        failure = {"error": "pregunta_invalida", "detalle": first_line(error)}
    except OutputParserException as error:
        failure = {"error": "salida_invalida", "detalle": first_line(error)}
    except PROVIDER_FAILURES as error:
        info = classify_sdk_error(error, provider)
        failure = {"error": info.kind, "detalle": info.message, "status_code": info.status_code}
    except Exception as error:
        failure = {"error": "inesperado", "detalle": f"{type(error).__name__}: {first_line(error)}"}
    else:
        return respuesta.model_dump_json(indent=2)
    return json.dumps({"pregunta": pregunta, **failure}, indent=2, ensure_ascii=False)


async def show_retrieval(rag_settings: RagSettings, preguntas: list[str]) -> None:
    store = open_vectorstore(rag_settings, local_embeddings(rag_settings))
    for pregunta in preguntas:
        candidatos = await search(store, pregunta, top_k=rag_settings.top_k)
        usados = relevant(candidatos, rag_settings.min_relevance)
        print(f"\n=== {pregunta} ===")
        for fragmento in candidatos:
            usado = "sí" if fragmento in usados else "no"
            seccion = fragmento.documento.metadata.get("seccion")
            print(f"{fragmento.relevancia:.3f}  usado={usado}  {fragmento.documento.id}  {seccion}")


async def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        settings, rag_settings = Settings(), RagSettings()
    except ValidationError as error:
        print(f"Configuración inválida en el .env:\n{error}")
        return 2
    preguntas = [args.pregunta] if args.pregunta else list(SAMPLE_QUESTIONS)
    embeddings = local_embeddings(rag_settings)
    try:
        ingest(rag_settings, embeddings)
        if args.solo_recuperar:
            await show_retrieval(rag_settings, preguntas)
            return 0
        provider = settings.preferred_provider(args.provider)
        if provider is None:
            print(
                "No hay API keys configuradas. Copiá .env.example a .env y completá "
                "OPENAI_API_KEY y/o ANTHROPIC_API_KEY (o usá --solo-recuperar)."
            )
            return 1
        chain = build_rag_chain(
            open_vectorstore(rag_settings, embeddings),
            build_chat_model(settings.config_for(provider)),
            provider=provider,
            retry=settings.retry_policy(),
            top_k=rag_settings.top_k,
            min_relevance=rag_settings.min_relevance,
        )
    except (IngestError, EmbeddingMismatchError, ConfigurationError) as error:
        print(error)
        return 1
    respuestas = await asyncio.gather(*(answer(chain, provider, p) for p in preguntas))
    print(f"\nProveedor: {provider}\n")
    print("\n\n".join(respuestas))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "openai", "anthropic", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.exit(asyncio.run(main(sys.argv[1:])))
