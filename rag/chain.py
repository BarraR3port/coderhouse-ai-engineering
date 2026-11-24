import logging
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, TypeAlias

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStore

from entity_pipeline import PipelineLogger
from entity_pipeline.chain import jitter_params, retry_condition
from llm_client import Provider, RetryPolicy
from rag.schemas import NO_LO_SE, Fuente, RespuestaDelModelo, RespuestaRAG, SolicitudRAG

logger = logging.getLogger(__name__)

RagChain: TypeAlias = Runnable[dict[str, str], RespuestaRAG]

SYSTEM_PROMPT = """Sos el asistente técnico interno de la plataforma Nimbus.
Respondé la PREGUNTA usando exclusivamente la información del CONTEXTO: fragmentos numerados \
de la documentación interna.

Reglas:
- No uses conocimiento propio ni completes con datos que no estén en el CONTEXTO.
- Si el CONTEXTO no alcanza para responder, la respuesta es exactamente "{no_lo_se}" y \
fragmentos_usados queda vacío.
- En fragmentos_usados poné los números de los fragmentos en los que te basaste.
- Respondé en español, en cuatro oraciones como máximo.

{format_instructions}"""

HUMAN_PROMPT = """CONTEXTO:
{contexto}

PREGUNTA: {pregunta}"""


@dataclass(frozen=True)
class Fragmento:
    documento: Document
    relevancia: float


def build_rag_chain(
    store: VectorStore,
    llm: BaseChatModel,
    *,
    provider: Provider,
    retry: RetryPolicy,
    top_k: int,
    min_relevance: float,
) -> RagChain:
    async def retrieve(pregunta: str) -> list[Fragmento]:
        candidatos = await search(store, pregunta, top_k=top_k)
        fragmentos = relevant(candidatos, min_relevance)
        logger.info(
            "Recuperados %d de %d fragmentos con relevancia >= %.2f: %s",
            len(fragmentos),
            len(candidatos),
            min_relevance,
            ", ".join(f"{f.documento.id} ({f.relevancia:.3f})" for f in candidatos) or "ninguno",
        )
        return fragmentos

    parser = PydanticOutputParser(pydantic_object=RespuestaDelModelo)
    answer_from_context = (
        RunnablePassthrough.assign(contexto=RunnableLambda(format_context, name="contexto"))
        | RunnablePassthrough.assign(salida=build_prompt(parser) | llm | parser)
        | RunnableLambda(cite_sources, name="citar_fuentes")
    ).with_retry(
        retry_if_exception_type=retry_condition(provider),
        wait_exponential_jitter=True,
        exponential_jitter_params=jitter_params(retry),
        stop_after_attempt=retry.max_attempts,
    )
    chain = RunnablePassthrough.assign(
        fragmentos=itemgetter("pregunta") | RunnableLambda(retrieve, name="recuperar")
    ) | RunnableBranch(
        (lambda inputs: not inputs["fragmentos"], RunnableLambda(no_context, name="sin_contexto")),
        answer_from_context,
    )
    return chain.with_config(run_name="rag_nimbus", callbacks=[PipelineLogger()])


async def search(store: VectorStore, pregunta: str, *, top_k: int) -> list[Fragmento]:
    scored = await store.asimilarity_search_with_relevance_scores(pregunta, k=top_k)
    return [Fragmento(documento, relevancia) for documento, relevancia in scored]


def relevant(fragmentos: list[Fragmento], min_relevance: float) -> list[Fragmento]:
    return [fragmento for fragmento in fragmentos if fragmento.relevancia >= min_relevance]


async def get_rag_response(chain: RagChain, pregunta: str) -> RespuestaRAG:
    solicitud = SolicitudRAG(pregunta=pregunta)
    return await chain.ainvoke({"pregunta": solicitud.pregunta})


def build_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    ).partial(no_lo_se=NO_LO_SE, format_instructions=parser.get_format_instructions())


def format_context(inputs: dict[str, Any]) -> str:
    return "\n\n".join(
        f"[{number}] ({describe_origin(fragmento.documento)})\n{fragmento.documento.page_content}"
        for number, fragmento in enumerate(inputs["fragmentos"], start=1)
    )


def describe_origin(documento: Document) -> str:
    seccion = documento.metadata.get("seccion")
    archivo = documento.metadata["archivo"]
    return f"{archivo} > {seccion}" if seccion else archivo


def cite_sources(inputs: dict[str, Any]) -> RespuestaRAG:
    salida: RespuestaDelModelo = inputs["salida"]
    fragmentos: list[Fragmento] = inputs["fragmentos"]
    if salida.respuesta == NO_LO_SE:
        logger.info("El modelo respondió que el contexto no alcanza")
        return RespuestaRAG(
            pregunta=inputs["pregunta"], respuesta=NO_LO_SE, encontrada=False, fuentes=[]
        )
    out_of_range = [n for n in salida.fragmentos_usados if not 1 <= n <= len(fragmentos)]
    if out_of_range:
        raise OutputParserException(
            f"el modelo citó fragmentos que no existen: {out_of_range} (hay {len(fragmentos)})"
        )
    usados = [fragmentos[n - 1] for n in dict.fromkeys(salida.fragmentos_usados)]
    return RespuestaRAG(
        pregunta=inputs["pregunta"],
        respuesta=salida.respuesta,
        encontrada=True,
        fuentes=[as_source(fragmento) for fragmento in usados],
    )


def as_source(fragmento: Fragmento) -> Fuente:
    documento = fragmento.documento
    return Fuente(
        archivo=documento.metadata["archivo"],
        seccion=documento.metadata.get("seccion"),
        fragmento_id=documento.id or f"{documento.metadata['archivo']}#?",
        relevancia=round(min(max(fragmento.relevancia, 0.0), 1.0), 3),
    )


def no_context(inputs: dict[str, Any]) -> RespuestaRAG:
    logger.info("Ningún fragmento supera la relevancia mínima: se responde sin llamar al modelo")
    return RespuestaRAG(
        pregunta=inputs["pregunta"], respuesta=NO_LO_SE, encontrada=False, fuentes=[]
    )
