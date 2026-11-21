"""Acceso compartido a ChromaDB persistente (lo usan ``ingest.py`` y ``retriever.py``).

La colección guarda en su metadata con qué modelo de embeddings se indexó y
una huella de los documentos. Con eso:

- la ingesta sabe si puede saltarse el re-indexado (nada cambió), y
- el retriever se niega a consultar con un modelo de embeddings distinto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

# Claves de la metadata de la colección.
META_EMBEDDINGS_PROVIDER = "embeddings_provider"
META_EMBEDDINGS_MODEL = "embeddings_model"
META_HUELLA = "huella"
META_CHUNK_SIZE = "chunk_size"
META_CHUNK_OVERLAP = "chunk_overlap"
META_ARCHIVOS = "archivos"

# Similitud coseno: la relevancia queda en [0, 1] y es fácil de interpretar.
CONFIGURACION_COLECCION: dict[str, Any] = {"hnsw": {"space": "cosine"}}


def abrir_cliente(persist_dir: Path) -> ClientAPI:
    """Cliente persistente de ChromaDB en ``persist_dir`` (sin telemetría)."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def obtener_coleccion(cliente: ClientAPI, nombre: str) -> Collection | None:
    """Devuelve la colección o ``None`` si todavía no existe."""
    try:
        return cliente.get_collection(nombre)
    except NotFoundError:
        return None
