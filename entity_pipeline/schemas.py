from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

MAX_INPUT_CHARS = 20_000
MAX_SUMMARY_CHARS = 600


class Criticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class SolicitudDeExtraccion(BaseModel):
    model_config = ConfigDict(frozen=True)

    texto: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_INPUT_CHARS)
    ]


class ExtraccionTecnica(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tecnologias: list[str] = Field(
        description=(
            "Tecnologías que nombra el texto: lenguajes, frameworks, bases de datos, colas, "
            "servicios o herramientas, con su nombre propio (por ejemplo PostgreSQL o Kubernetes)"
        )
    )
    nivel_de_criticidad: Criticidad = Field(
        description=(
            "alta si hay una caída, pérdida de datos o riesgo de seguridad; media si hay "
            "degradación o errores que se recuperan; baja si es informativo o no hay impacto"
        )
    )
    resumen_tecnico: str = Field(
        description="Una o dos oraciones en español que expliquen qué pasa a nivel técnico"
    )

    @field_validator("tecnologias")
    @classmethod
    def at_least_one_distinct_technology(cls, tecnologias: list[str]) -> list[str]:
        distinct: dict[str, str] = {}
        for name in (name.strip() for name in tecnologias):
            if name:
                distinct.setdefault(name.casefold(), name)
        if not distinct:
            raise ValueError("el texto tiene que nombrar al menos una tecnología")
        return list(distinct.values())

    @field_validator("resumen_tecnico")
    @classmethod
    def concise_summary(cls, resumen: str) -> str:
        resumen = resumen.strip()
        if not resumen:
            raise ValueError("el resumen técnico no puede estar vacío")
        if len(resumen) > MAX_SUMMARY_CHARS:
            raise ValueError(f"el resumen técnico supera los {MAX_SUMMARY_CHARS} caracteres")
        return resumen
