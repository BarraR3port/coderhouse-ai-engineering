from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NO_LO_SE = "No lo sé"
MAX_QUESTION_CHARS = 1_000


def is_no_lo_se(text: str) -> bool:
    return text.strip().rstrip(".").casefold() == NO_LO_SE.casefold()


class SolicitudRAG(BaseModel):
    model_config = ConfigDict(frozen=True)

    pregunta: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=3, max_length=MAX_QUESTION_CHARS)
    ]


class RespuestaDelModelo(BaseModel):
    respuesta: str = Field(
        description=(
            f'Respuesta en español basada solo en el CONTEXTO, o exactamente "{NO_LO_SE}" '
            "si el CONTEXTO no alcanza para responder"
        )
    )
    fragmentos_usados: list[int] = Field(
        default_factory=list,
        description=(
            "Números de los fragmentos del CONTEXTO en los que se basa la respuesta; "
            f'lista vacía si la respuesta es "{NO_LO_SE}"'
        ),
    )

    @model_validator(mode="after")
    def citations_match_the_answer(self) -> Self:
        if is_no_lo_se(self.respuesta):
            self.respuesta = NO_LO_SE
            self.fragmentos_usados = []
        elif not self.fragmentos_usados:
            raise ValueError("una respuesta basada en el CONTEXTO tiene que citar sus fragmentos")
        return self


class Fuente(BaseModel):
    archivo: str
    seccion: str | None = None
    fragmento_id: str
    relevancia: float = Field(ge=0.0, le=1.0, description="similitud coseno con la pregunta")


class RespuestaRAG(BaseModel):
    pregunta: str
    respuesta: str
    encontrada: bool
    fuentes: list[Fuente]

    @model_validator(mode="after")
    def sources_match_the_answer(self) -> Self:
        if self.encontrada and not self.fuentes:
            raise ValueError("una respuesta encontrada tiene que traer sus fuentes")
        if not self.encontrada and (self.fuentes or self.respuesta != NO_LO_SE):
            raise ValueError(f'sin información en el contexto la respuesta es "{NO_LO_SE}"')
        return self
