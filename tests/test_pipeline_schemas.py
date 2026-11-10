import pytest
from pydantic import ValidationError

from entity_pipeline import Criticidad, ExtraccionTecnica, SolicitudDeExtraccion
from entity_pipeline.schemas import MAX_INPUT_CHARS, MAX_SUMMARY_CHARS


def extraction(**overrides: object) -> ExtraccionTecnica:
    fields = {
        "tecnologias": ["FastAPI", "PostgreSQL"],
        "nivel_de_criticidad": "alta",
        "resumen_tecnico": "La base rechaza conexiones nuevas.",
    } | overrides
    return ExtraccionTecnica.model_validate(fields)


def test_extraction_parses_criticality_into_the_enum() -> None:
    assert extraction().nivel_de_criticidad is Criticidad.ALTA


def test_technologies_are_trimmed_and_deduplicated_ignoring_case() -> None:
    result = extraction(tecnologias=[" Kafka ", "kafka", "", "Redis", "REDIS"])

    assert result.tecnologias == ["Kafka", "Redis"]


@pytest.mark.parametrize("tecnologias", [[], ["", "   "]])
def test_at_least_one_technology_is_required(tecnologias: list[str]) -> None:
    with pytest.raises(ValidationError, match="al menos una tecnología"):
        extraction(tecnologias=tecnologias)


def test_criticality_outside_the_enum_is_rejected() -> None:
    with pytest.raises(ValidationError, match="nivel_de_criticidad"):
        extraction(nivel_de_criticidad="urgente")


def test_summary_is_trimmed() -> None:
    assert extraction(resumen_tecnico="  Cae el checkout.  ").resumen_tecnico == "Cae el checkout."


@pytest.mark.parametrize(
    ("resumen", "message"),
    [("   ", "no puede estar vacío"), ("x" * (MAX_SUMMARY_CHARS + 1), "supera los")],
)
def test_summary_must_be_present_and_concise(resumen: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        extraction(resumen_tecnico=resumen)


def test_fields_outside_the_contract_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        extraction(severidad="alta")


def test_request_strips_the_input_text() -> None:
    assert SolicitudDeExtraccion(texto="  log de error  ").texto == "log de error"


@pytest.mark.parametrize("texto", ["", "   ", "x" * (MAX_INPUT_CHARS + 1)])
def test_request_rejects_blank_or_huge_text(texto: str) -> None:
    with pytest.raises(ValidationError):
        SolicitudDeExtraccion(texto=texto)
