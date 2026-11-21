"""Dobles de prueba para correr los tests sin API key ni red.

- ``EmbeddingsPalabras``: embeddings deterministas por bolsa de palabras. No
  son "semánticos", pero una pregunta con las mismas palabras que un fragmento
  queda cerca de él, así que la búsqueda de similitud es verificable.
- ``ModeloGuionado``: modelo de chat que devuelve respuestas guionadas (o
  calculadas a partir del prompt) para simular JSON válido, mal formado,
  truncado, etc.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever

DIMENSIONES = 256


class EmbeddingsPalabras(Embeddings):
    """Vector = conteo de palabras (sin tildes, >3 letras) en buckets por hash."""

    def __init__(self, falla_en_llamada: int | None = None) -> None:
        self.llamadas = 0
        self.textos_embebidos = 0
        self.falla_en_llamada = falla_en_llamada

    def _vector(self, texto: str) -> list[float]:
        plano = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
        vector = [0.0] * DIMENSIONES
        for palabra in re.findall(r"[a-z0-9]{4,}", plano):
            bucket = int(hashlib.md5(palabra.encode()).hexdigest(), 16) % DIMENSIONES
            vector[bucket] += 1.0
        norma = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norma for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.llamadas += 1
        if self.falla_en_llamada is not None and self.llamadas >= self.falla_en_llamada:
            raise ConnectionError("API de embeddings caída (simulada)")
        self.textos_embebidos += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class RetrieverFijo(BaseRetriever):
    """Retriever que siempre devuelve los mismos documentos y cuenta sus llamadas."""

    documentos: list[Document]
    llamadas: int = 0

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        self.llamadas += 1
        return [d.model_copy(deep=True) for d in self.documentos]


class ModeloGuionado(BaseChatModel):
    """Devuelve, en orden, los mensajes de ``respuestas`` (repite el último).

    Si se pasa ``responder``, la respuesta se calcula a partir de los mensajes
    recibidos (útil para pruebas concurrentes).
    """

    respuestas: list[AIMessage] = []
    responder: Callable[[list[BaseMessage]], AIMessage] | None = None
    llamadas: int = 0
    prompts_recibidos: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "modelo-guionado"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.llamadas += 1
        self.prompts_recibidos.append(list(messages))
        if self.responder is not None:
            mensaje = self.responder(list(messages))
        else:
            indice = min(self.llamadas - 1, len(self.respuestas) - 1)
            mensaje = self.respuestas[indice].model_copy(deep=True)
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


def respuesta_json(
    respuesta: str = "Se compensa con 1,5 días libres.",
    encontrada: bool = True,
    citados: list[str] | None = None,
    *,
    finish_reason: str = "stop",
    con_bloque_markdown: bool = False,
) -> AIMessage:
    """Mensaje con el JSON que espera ``PydanticOutputParser``."""
    contenido = json.dumps(
        {
            "respuesta": respuesta,
            "encontrada_en_contexto": encontrada,
            "fragmentos_citados": ["F1"] if citados is None else citados,
        },
        ensure_ascii=False,
    )
    if con_bloque_markdown:
        contenido = f"```json\n{contenido}\n```"
    return mensaje_texto(contenido, finish_reason=finish_reason)


def respuesta_no_se() -> AIMessage:
    return respuesta_json("No lo sé.", encontrada=False, citados=[])


def mensaje_texto(contenido: str, *, finish_reason: str = "stop") -> AIMessage:
    return AIMessage(content=contenido, response_metadata={"finish_reason": finish_reason})
