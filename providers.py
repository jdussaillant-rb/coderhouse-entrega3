"""Fábricas de modelos: embeddings y LLM de chat, según la configuración.

Cada proveedor es una función registrada en ``PROVEEDORES_EMBEDDINGS`` o
``PROVEEDORES_LLM``. Para sumar uno nuevo basta con escribir su fábrica y
registrarla (ver la sección "Agregar un proveedor" del README).

Importante: el modelo de embeddings con el que se indexa queda guardado en la
metadata de la colección de ChromaDB. ``retriever.py`` se niega a consultar con
un modelo distinto, porque los vectores de dos modelos no son comparables.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from config import Settings

logger = logging.getLogger("rag.providers")

T = TypeVar("T")


class ProveedorNoSoportadoError(ValueError):
    """El proveedor pedido no está registrado."""


class FaltaApiKeyError(RuntimeError):
    """Falta la API key del proveedor elegido."""


@dataclass(frozen=True)
class Proveedor(Generic[T]):
    """Un proveedor registrado: su fábrica, su modelo por defecto y su API key."""

    fabrica: Callable[[str, Settings], T]
    modelo_por_defecto: str
    variable_api_key: str | None = None


def _exigir_api_key(variable: str | None) -> None:
    if variable and not os.getenv(variable):
        raise FaltaApiKeyError(f"Falta {variable} en el entorno o en .env (ver .env.example).")


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def _openai_embeddings(modelo: str, settings: Settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=modelo, max_retries=2, timeout=settings.llm_timeout)


PROVEEDORES_EMBEDDINGS: dict[str, Proveedor[Embeddings]] = {
    # Anthropic no ofrece modelos de embeddings: por eso el único proveedor
    # registrado es OpenAI, aunque el LLM de chat sea Claude.
    "openai": Proveedor(_openai_embeddings, "text-embedding-3-small", "OPENAI_API_KEY"),
}


def resolver_modelo_embeddings(settings: Settings) -> tuple[str, str]:
    """Devuelve ``(proveedor, modelo)`` de embeddings, aplicando defaults."""
    proveedor = settings.embeddings_provider.lower()
    registro = _buscar(PROVEEDORES_EMBEDDINGS, proveedor, "EMBEDDINGS_PROVIDER")
    return proveedor, settings.embeddings_model or registro.modelo_por_defecto


def build_embeddings(settings: Settings) -> Embeddings:
    """Crea el cliente de embeddings configurado (el mismo para indexar y consultar)."""
    proveedor, modelo = resolver_modelo_embeddings(settings)
    registro = PROVEEDORES_EMBEDDINGS[proveedor]
    _exigir_api_key(registro.variable_api_key)
    logger.info("Embeddings: %s (%s)", proveedor, modelo)
    return registro.fabrica(modelo, settings)


# --------------------------------------------------------------------------- #
# LLM de chat
# --------------------------------------------------------------------------- #
def _openai_chat(modelo: str, settings: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=modelo,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=2,  # red y rate limit; los errores de formato los reintenta la cadena
    )


def _anthropic_chat(modelo: str, settings: Settings) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=modelo,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=2,
    )


PROVEEDORES_LLM: dict[str, Proveedor[BaseChatModel]] = {
    "openai": Proveedor(_openai_chat, "gpt-4o-mini", "OPENAI_API_KEY"),
    "anthropic": Proveedor(_anthropic_chat, "claude-haiku-5-5", "ANTHROPIC_API_KEY"),
}


def build_llm(settings: Settings) -> BaseChatModel:
    """Crea el modelo de chat configurado en ``LLM_PROVIDER`` / ``LLM_MODEL``."""
    proveedor = settings.llm_provider.lower()
    registro = _buscar(PROVEEDORES_LLM, proveedor, "LLM_PROVIDER")
    modelo = settings.llm_model or registro.modelo_por_defecto
    _exigir_api_key(registro.variable_api_key)
    logger.info("LLM: %s (%s)", proveedor, modelo)
    return registro.fabrica(modelo, settings)


def _buscar(registro: dict[str, Proveedor[T]], nombre: str, variable: str) -> Proveedor[T]:
    try:
        return registro[nombre]
    except KeyError:
        disponibles = ", ".join(sorted(registro))
        raise ProveedorNoSoportadoError(
            f"{variable}='{nombre}' no está soportado. Opciones: {disponibles}."
        ) from None
