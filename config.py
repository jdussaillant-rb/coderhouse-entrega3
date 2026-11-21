"""Configuración del sistema RAG, validada con Pydantic.

Todos los parámetros salen de variables de entorno (o de un ``.env``), así que
las claves de API nunca quedan en el código. ``Settings`` valida los valores al
cargarlos: un ``RAG_TOP_K`` fuera de rango o un solapamiento mayor que el
tamaño de fragmento fallan al arrancar, no en mitad de una consulta.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Defaults del proyecto. Los proveedores y modelos se resuelven en providers.py.
DEFAULT_DATA_DIR = "data"
DEFAULT_PERSIST_DIR = "vectorstore"
DEFAULT_COLLECTION = "nimbus_docs"

# Consigna: chunking de mínimo 500 tokens con 50 de solapamiento.
MIN_CHUNK_TOKENS = 500
MIN_CHUNK_OVERLAP = 50
# Consigna: top_k entre 3 y 5 (evita el "contexto infinito" y el "Lost in the Middle").
MIN_TOP_K = 3
MAX_TOP_K = 5

# Variable de entorno -> campo de Settings.
VARIABLES_DE_ENTORNO: dict[str, str] = {
    "RAG_DATA_DIR": "data_dir",
    "RAG_PERSIST_DIR": "persist_dir",
    "RAG_COLLECTION": "collection_name",
    "RAG_CHUNK_SIZE": "chunk_size",
    "RAG_CHUNK_OVERLAP": "chunk_overlap",
    "RAG_TOP_K": "top_k",
    "EMBEDDINGS_PROVIDER": "embeddings_provider",
    "EMBEDDINGS_MODEL": "embeddings_model",
    "LLM_PROVIDER": "llm_provider",
    "LLM_MODEL": "llm_model",
    "LLM_TEMPERATURE": "llm_temperature",
    "LLM_MAX_TOKENS": "llm_max_tokens",
    "LLM_TIMEOUT": "llm_timeout",
}


class ConfiguracionInvalidaError(ValueError):
    """Alguna variable de entorno tiene un valor no permitido."""


class Settings(BaseModel):
    """Parámetros del sistema. Inmutable una vez creado."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    # --- Rutas ----------------------------------------------------------------
    data_dir: Path = Field(default=Path(DEFAULT_DATA_DIR), description="Carpeta con los .md/.txt a indexar.")
    persist_dir: Path = Field(default=Path(DEFAULT_PERSIST_DIR), description="Carpeta de ChromaDB persistente.")
    collection_name: str = Field(default=DEFAULT_COLLECTION, min_length=3, max_length=63)

    # --- Chunking -------------------------------------------------------------
    chunk_size: int = Field(default=MIN_CHUNK_TOKENS, ge=MIN_CHUNK_TOKENS, le=4_000, description="Tokens por fragmento.")
    chunk_overlap: int = Field(default=MIN_CHUNK_OVERLAP, ge=MIN_CHUNK_OVERLAP, description="Tokens de solapamiento.")

    # --- Recuperación ---------------------------------------------------------
    top_k: int = Field(default=4, ge=MIN_TOP_K, le=MAX_TOP_K, description="Fragmentos que recibe el LLM.")

    # --- Embeddings -----------------------------------------------------------
    embeddings_provider: str = Field(default="openai", min_length=1)
    embeddings_model: str | None = Field(default=None, description="None = modelo por defecto del proveedor.")

    # --- LLM ------------------------------------------------------------------
    llm_provider: str = Field(default="openai", min_length=1)
    llm_model: str | None = Field(default=None, description="None = modelo por defecto del proveedor.")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1_024, ge=64, le=16_000)
    llm_timeout: float = Field(default=30.0, gt=0, le=300)

    @model_validator(mode="after")
    def _validar_solapamiento(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) debe ser menor que chunk_size ({self.chunk_size})"
            )
        return self

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        """Crea la configuración desde el entorno; ``overrides`` tiene prioridad.

        Las variables vacías se ignoran (se usa el default).
        """
        load_dotenv()
        valores: dict[str, Any] = {}
        for variable, campo in VARIABLES_DE_ENTORNO.items():
            valor = os.getenv(variable)
            if valor is not None and valor.strip():
                valores[campo] = valor.strip()
        valores.update({k: v for k, v in overrides.items() if v is not None})
        try:
            return cls(**valores)
        except ValidationError as exc:
            errores = "; ".join(
                f"{_nombre_variable(str(e['loc'][0])) if e['loc'] else 'config'}: {e['msg']}"
                for e in exc.errors()
            )
            raise ConfiguracionInvalidaError("Configuración inválida -> " + errores) from exc


def _nombre_variable(campo: str) -> str:
    """Traduce un campo de Settings a su variable de entorno, para errores legibles."""
    for variable, nombre in VARIABLES_DE_ENTORNO.items():
        if nombre == campo:
            return variable
    return campo
