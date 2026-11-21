"""Contratos de salida del sistema RAG.

Hay dos modelos:

- ``RespuestaLLM``: lo que el LLM debe devolver y ``PydanticOutputParser``
  valida. Es deliberadamente simple: el texto de la respuesta, si se encontró en
  el contexto y los IDs de los fragmentos citados (``F1``, ``F2``...).
- ``RespuestaRAG``: lo que devuelve ``get_rag_response``. Se arma en el código
  a partir de ``RespuestaLLM``, reemplazando cada ID por la referencia real
  (archivo, sección, relevancia). Así las referencias nunca las inventa el
  modelo: si cita un ID que no se recuperó, se descarta.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RESPUESTA_NO_SE = "No lo sé."
MAX_LARGO_RESPUESTA = 2_000
_PATRON_ID = re.compile(r"^F\d+$")


class RespuestaLLM(BaseModel):
    """Respuesta del modelo basada exclusivamente en el CONTEXTO."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    respuesta: str = Field(
        min_length=1,
        max_length=MAX_LARGO_RESPUESTA,
        description=(
            "Respuesta en español basada solo en el CONTEXTO. Si el CONTEXTO no "
            f"contiene la información, exactamente: '{RESPUESTA_NO_SE}'"
        ),
    )
    encontrada_en_contexto: bool = Field(
        description="true si la respuesta está respaldada por el CONTEXTO; false si no lo está.",
    )
    fragmentos_citados: list[str] = Field(
        default_factory=list,
        description=(
            "IDs de los fragmentos del CONTEXTO que respaldan la respuesta, por ejemplo "
            "['F1', 'F3']. Lista vacía si encontrada_en_contexto es false."
        ),
    )

    @field_validator("fragmentos_citados", mode="before")
    @classmethod
    def normalizar_ids(cls, valores: object) -> object:
        """Acepta '[F1]', 'f1' o ' F1 ' y elimina duplicados conservando el orden."""
        if not isinstance(valores, list):
            return valores
        limpios: list[str] = []
        for valor in valores:
            if not isinstance(valor, str):
                raise ValueError("cada fragmento citado debe ser un string como 'F1'")
            normalizado = valor.strip().strip("[]").strip().upper()
            if not _PATRON_ID.match(normalizado):
                raise ValueError(f"ID de fragmento inválido: '{valor}' (se espera 'F1', 'F2'...)")
            if normalizado not in limpios:
                limpios.append(normalizado)
        return limpios

    @model_validator(mode="after")
    def coherencia(self) -> RespuestaLLM:
        """Una respuesta "encontrada" debe citar fuentes; una "no encontrada" es "No lo sé."."""
        if self.encontrada_en_contexto:
            if not self.fragmentos_citados:
                raise ValueError(
                    "si encontrada_en_contexto es true, fragmentos_citados no puede estar vacío"
                )
        else:
            self.respuesta = RESPUESTA_NO_SE
            self.fragmentos_citados = []
        return self


class Referencia(BaseModel):
    """Fragmento recuperado de ChromaDB que respalda la respuesta."""

    id: str = Field(description="ID del fragmento en el prompt (F1, F2...).")
    fuente: str = Field(description="Archivo de origen dentro de /data.")
    seccion: str = Field(default="", description="Secciones del documento que abarca el fragmento.")
    fragmento: int = Field(ge=0, description="Posición del fragmento dentro del archivo.")
    relevancia: float | None = Field(default=None, description="Similitud coseno con la pregunta (0 a 1).")
    extracto: str = Field(description="Comienzo del fragmento, para verificar la cita.")


class RespuestaRAG(BaseModel):
    """Resultado final de ``get_rag_response``: texto y referencias verificadas."""

    pregunta: str
    respuesta: str
    encontrada_en_contexto: bool
    referencias: list[Referencia] = Field(default_factory=list)
