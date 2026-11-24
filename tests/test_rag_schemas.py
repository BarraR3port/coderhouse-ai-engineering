import pytest
from pydantic import ValidationError

from rag import NO_LO_SE, Fuente, RespuestaRAG, SolicitudRAG
from rag.schemas import MAX_QUESTION_CHARS

FUENTE = Fuente(archivo="a.md", seccion="S", fragmento_id="a.md#000", relevancia=0.5)


@pytest.mark.parametrize("pregunta", ["ok", "x" * (MAX_QUESTION_CHARS + 1)])
def test_questions_too_short_or_too_long_are_rejected(pregunta: str) -> None:
    with pytest.raises(ValidationError):
        SolicitudRAG(pregunta=pregunta)


def test_a_question_is_stripped() -> None:
    assert SolicitudRAG(pregunta="  ¿Qué es SEV1?  ").pregunta == "¿Qué es SEV1?"


def test_a_found_answer_needs_sources() -> None:
    with pytest.raises(ValidationError, match="fuentes"):
        RespuestaRAG(pregunta="p", respuesta="r", encontrada=True, fuentes=[])


@pytest.mark.parametrize(
    ("respuesta", "fuentes"),
    [(NO_LO_SE, [FUENTE]), ("Una respuesta inventada", [])],
)
def test_an_answer_not_found_is_no_lo_se_without_sources(
    respuesta: str, fuentes: list[Fuente]
) -> None:
    with pytest.raises(ValidationError, match=NO_LO_SE):
        RespuestaRAG(pregunta="p", respuesta=respuesta, encontrada=False, fuentes=fuentes)
