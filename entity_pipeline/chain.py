import logging
from collections.abc import Callable
from typing import Any
from uuid import UUID

import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import ModelRefusalError, OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from entity_pipeline.schemas import ExtraccionTecnica, SolicitudDeExtraccion
from llm_client import LLMConfig, Provider, RetryPolicy, Settings
from llm_client.errors import PROVIDER_FAILURES, classify_sdk_error

logger = logging.getLogger(__name__)

OUTPUT_ERRORS = (ValidationError, OutputParserException, openai.LengthFinishReasonError)
REFUSAL_ERRORS = (ModelRefusalError, openai.ContentFilterFinishReasonError)
TRUNCATION_REASONS = frozenset({"length", "max_tokens", "model_context_window_exceeded"})
RETRY_TAG_PREFIX = "retry:attempt:"

type ExtractionChain = Runnable[dict[str, str], ExtraccionTecnica]

SYSTEM_PROMPT = """Sos un ingeniero de software senior que analiza textos técnicos: \
descripciones de arquitectura, reportes de incidentes y logs de error.
Extraé la información solo a partir del texto, sin agregar tecnologías que no aparezcan en él.

{format_instructions}"""


class IncompleteOutputError(OutputParserException):
    pass


def build_extraction_chain(
    settings: Settings,
    provider: Provider | None = None,
    *,
    base_url: str | None = None,
) -> ExtractionChain:
    provider = provider or settings.llm_provider
    retry = settings.retry_policy()
    llm = build_chat_model(settings.config_for(provider), base_url=base_url)
    # json_schema también en Anthropic: forzar tool_choice da 400 en Claude Opus 5.5.
    structured_llm = llm.with_structured_output(
        ExtraccionTecnica, method="json_schema", include_raw=True
    )
    validated_llm = structured_llm | RunnableLambda(checked_extraction, name="validar_extraccion")
    resilient_llm = validated_llm.with_retry(
        retry_if_exception_type=retry_condition(provider),
        wait_exponential_jitter=True,
        exponential_jitter_params=jitter_params(retry),
        stop_after_attempt=retry.max_attempts,
    )
    chain = build_prompt() | resilient_llm
    return chain.with_config(run_name="extraccion_tecnica", callbacks=[PipelineLogger()])


async def extract(chain: ExtractionChain, texto: str) -> ExtraccionTecnica:
    request = SolicitudDeExtraccion(texto=texto)
    return await chain.ainvoke(request.model_dump())


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{texto}")]
    ).partial(format_instructions=format_instructions(ExtraccionTecnica))


def format_instructions(schema: type[BaseModel]) -> str:
    fields = schema.model_fields.items()
    lines = ["- " + name + ": " + str(field.description) for name, field in fields]
    return "\n".join(["Completá cada campo así:", *lines])


def build_chat_model(config: LLMConfig, *, base_url: str | None = None) -> BaseChatModel:
    match config.provider:
        case Provider.OPENAI:
            return ChatOpenAI(
                model=config.model,
                api_key=config.api_key,
                base_url=base_url,
                temperature=0,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds,
                max_retries=0,
            )
        case Provider.ANTHROPIC:
            return ChatAnthropic(
                model=config.model,
                api_key=config.api_key,
                base_url=base_url,
                effort=config.effort,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds,
                max_retries=0,
            )


def checked_extraction(output: dict[str, Any]) -> ExtraccionTecnica:
    if (reason := finish_reason(output["raw"])) in TRUNCATION_REASONS:
        error = IncompleteOutputError(f"el objeto quedó incompleto (terminó por {reason})")
        log_failed_output(error)
        raise error
    if (error := output["parsing_error"]) is not None:
        log_failed_output(error)
        raise error
    extraction = output["parsed"]
    if not isinstance(extraction, ExtraccionTecnica):
        raise OutputParserException("el modelo no devolvió un objeto ExtraccionTecnica")
    logger.info(
        "Salida validada: %d tecnologías, criticidad %s",
        len(extraction.tecnologias),
        extraction.nivel_de_criticidad,
    )
    return extraction


def retry_condition(provider: Provider) -> Callable[[BaseException], bool]:
    def should_retry(error: BaseException) -> bool:
        if isinstance(error, OUTPUT_ERRORS):
            return True
        if isinstance(error, PROVIDER_FAILURES):
            return classify_sdk_error(error, provider).retryable
        return False

    return should_retry


def jitter_params(retry: RetryPolicy) -> dict[str, float]:
    return {
        "initial": retry.base_delay_seconds,
        "max": retry.max_delay_seconds,
        "jitter": retry.base_delay_seconds,
    }


class PipelineLogger(BaseCallbackHandler):
    run_inline = True

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if (attempt := retry_attempt(tags)) is not None:
            logger.info("Reintentando la extracción: intento %d", attempt)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        logger.info("Llamando a %s", (metadata or {}).get("ls_model_name", "el modelo"))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        generation = response.generations[0][0]
        if not isinstance(generation, ChatGeneration):
            return
        usage = generation.message.usage_metadata or {}
        logger.info(
            "Respuesta recibida: fin=%s, tokens=%s entrada / %s salida",
            finish_reason(generation.message),
            usage.get("input_tokens", "?"),
            usage.get("output_tokens", "?"),
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if isinstance(error, OUTPUT_ERRORS + REFUSAL_ERRORS):
            log_failed_output(error)
        else:
            logger.warning("Falló la llamada al modelo: %s", type(error).__name__)


def retry_attempt(tags: list[str] | None) -> int | None:
    for tag in tags or []:
        if tag.startswith(RETRY_TAG_PREFIX):
            return int(tag.removeprefix(RETRY_TAG_PREFIX))
    return None


def finish_reason(message: BaseMessage) -> str | None:
    metadata = message.response_metadata
    return metadata.get("finish_reason") or metadata.get("stop_reason")


def log_failed_output(error: BaseException) -> None:
    match error:
        case IncompleteOutputError() | openai.LengthFinishReasonError():
            logger.warning("El modelo cortó la respuesta: %s", describe_output_error(error))
        case ModelRefusalError() | openai.ContentFilterFinishReasonError():
            logger.warning("El modelo no respondió: %s", describe_output_error(error))
        case _:
            logger.warning("La salida no cumple el esquema: %s", describe_output_error(error))


def describe_output_error(error: BaseException) -> str:
    match error:
        case ValidationError():
            return "; ".join(map(describe_validation_detail, error.errors()))
        case OutputParserException() if isinstance(error.__cause__, ValidationError):
            return describe_output_error(error.__cause__)
        case openai.LengthFinishReasonError():
            return "el objeto quedó incompleto (terminó por length)"
        case openai.ContentFilterFinishReasonError():
            return "el filtro de contenido del proveedor bloqueó la respuesta"
        case ModelRefusalError():
            return "el modelo se negó a responder: " + first_line(error)
        case _:
            return first_line(error)


def describe_validation_detail(detail: ErrorDetails) -> str:
    location = ".".join(map(str, detail["loc"]))
    return f"{location}: {detail['msg']}" if location else detail["msg"]


def first_line(error: BaseException) -> str:
    text = str(error).strip()
    return text.splitlines()[0] if text else type(error).__name__
