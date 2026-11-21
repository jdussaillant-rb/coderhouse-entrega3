"""Módulo de ingesta: lee ``/data``, fragmenta y persiste en ChromaDB.

Flujo::

    data/*.md|*.txt ─► cargar_documentos ─► fragmentar (500 tokens, 50 de solapamiento)
                    ─► embeddings ─► ChromaDB persistente (./vectorstore)

Antes de indexar se calcula una huella (SHA-256) de los archivos, de los
parámetros de chunking y del modelo de embeddings. Si coincide con la guardada
en la colección, no se vuelve a indexar: no se gastan tokens de embeddings en
documentos que ya están en la base. ``--force`` reconstruye igual.

Uso::

    python ingest.py            # indexa solo si algo cambió
    python ingest.py --force    # reconstruye la colección
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import ConfiguracionInvalidaError, Settings
from providers import FaltaApiKeyError, build_embeddings, resolver_modelo_embeddings
from store import (
    CONFIGURACION_COLECCION,
    META_ARCHIVOS,
    META_CHUNK_OVERLAP,
    META_CHUNK_SIZE,
    META_EMBEDDINGS_MODEL,
    META_EMBEDDINGS_PROVIDER,
    META_HUELLA,
    abrir_cliente,
    obtener_coleccion,
)

logger = logging.getLogger("rag.ingest")

EXTENSIONES_SOPORTADAS = frozenset({".md", ".txt"})
# Tokenizador de los modelos de OpenAI; sirve para contar tokens de forma realista.
ENCODING_TIKTOKEN = "cl100k_base"
# Separadores (regex). El primer nivel corta en saltos de línea y en fin de
# oración, así las unidades que se combinan son chicas (10 a 60 tokens) y el
# solapamiento de 50 tokens funciona de verdad: el splitter solo puede solapar
# unidades completas, y con párrafos o secciones enteras el solapamiento real
# quedaba en 0. Las palabras y los caracteres son el último recurso.
SEPARADORES = [r"\n+|(?<=[.!?:])\s+", r"\s+", ""]
# Si cambia la lógica de fragmentación, cambia la huella y se re-indexa.
VERSION_FRAGMENTADOR = "v3"

_PATRON_TITULO = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


class SinDocumentosError(FileNotFoundError):
    """La carpeta de datos no existe o no tiene archivos .md/.txt con contenido."""


@dataclass(frozen=True)
class ResultadoIngesta:
    """Resumen de una ejecución de la ingesta."""

    estado: Literal["indexado", "sin_cambios"]
    archivos: int
    fragmentos: int
    coleccion: str


# --------------------------------------------------------------------------- #
# 1. Carga
# --------------------------------------------------------------------------- #
def listar_archivos(data_dir: Path) -> list[Path]:
    """Archivos ``.md``/``.txt`` de ``data_dir`` (recursivo), en orden estable."""
    if not data_dir.is_dir():
        raise SinDocumentosError(f"No existe la carpeta de datos: {data_dir}")
    archivos = sorted(
        p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES_SOPORTADAS
    )
    if not archivos:
        raise SinDocumentosError(f"No hay archivos .md o .txt en {data_dir}")
    return archivos


def cargar_documentos(archivos: list[Path], data_dir: Path) -> list[Document]:
    """Lee cada archivo como un ``Document`` con su ruta relativa en ``source``."""
    documentos: list[Document] = []
    for archivo in archivos:
        texto = archivo.read_text(encoding="utf-8").strip()
        if not texto:
            logger.warning("Se omite %s: está vacío", archivo)
            continue
        fuente = archivo.relative_to(data_dir).as_posix()
        documentos.append(Document(page_content=texto, metadata={"source": fuente}))
    if not documentos:
        raise SinDocumentosError(f"Todos los archivos de {data_dir} están vacíos")
    return documentos


# --------------------------------------------------------------------------- #
# 2. Fragmentación (chunking)
# --------------------------------------------------------------------------- #
def crear_fragmentador(settings: Settings) -> RecursiveCharacterTextSplitter:
    """``RecursiveCharacterTextSplitter`` que mide el tamaño en tokens, no en caracteres."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_TIKTOKEN,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARADORES,
        is_separator_regex=True,
        keep_separator="start",
        strip_whitespace=True,
        # add_start_index no se usa: con longitud en tokens, LangChain resta el
        # solapamiento (en tokens) a una posición en caracteres y devuelve -1.
        # La posición se calcula en fragmentar().
    )


def fragmentar(documentos: list[Document], settings: Settings) -> list[Document]:
    """Divide los documentos y enriquece la metadata de cada fragmento.

    Metadata de cada fragmento:
        - ``source``: archivo de origen.
        - ``chunk``: posición del fragmento dentro de su archivo (0, 1, 2...).
        - ``section``: secciones Markdown que abarca (ej. "2. Roles › 3. Guardia").
        - ``start_index``: posición (en caracteres) del fragmento en el texto original.

    Fragmentos consecutivos comparten hasta ``chunk_overlap`` tokens: las
    últimas oraciones de uno se repiten al comienzo del siguiente.
    """
    fragmentador = crear_fragmentador(settings)
    fragmentos: list[Document] = []
    for documento in documentos:
        texto = documento.page_content
        titulos = _titulos(texto)
        desde = 0
        for indice, fragmento in enumerate(fragmentador.split_documents([documento])):
            inicio = texto.find(fragmento.page_content, desde)
            desde = max(inicio, 0) + 1
            fragmento.metadata["start_index"] = inicio
            fragmento.metadata["chunk"] = indice
            fin = inicio + len(fragmento.page_content)
            fragmento.metadata["section"] = _secciones(titulos, inicio, fin)
            fragmentos.append(fragmento)
    return fragmentos


def _titulos(texto: str) -> list[tuple[int, int, str]]:
    """``(posición, nivel, título)`` de cada encabezado Markdown del texto."""
    return [(m.start(), len(m.group(1)), m.group(2).strip()) for m in _PATRON_TITULO.finditer(texto)]


def _secciones(titulos: list[tuple[int, int, str]], inicio: int, fin: int) -> str:
    """Secciones que abarca el fragmento ``[inicio, fin)``, separadas por " › ".

    Incluye el título vigente al comienzo del fragmento y los que aparecen
    dentro de él. El título del documento (nivel 1) solo se usa si no hay otro.
    """
    vigente: tuple[int, int, str] | None = None
    dentro: list[tuple[int, int, str]] = []
    for titulo in titulos:
        posicion = titulo[0]
        if posicion <= inicio:
            vigente = titulo
        elif posicion < fin:
            dentro.append(titulo)
    candidatos = ([vigente] if vigente else []) + dentro
    sin_nivel_1 = [t for t in candidatos if t[1] > 1]
    return " › ".join(t[2] for t in (sin_nivel_1 or candidatos))


def id_fragmento(fragmento: Document) -> str:
    """ID determinista (``archivo::n``): re-indexar no duplica fragmentos."""
    return f"{fragmento.metadata['source']}::{fragmento.metadata['chunk']}"


# --------------------------------------------------------------------------- #
# 3. Persistencia en ChromaDB
# --------------------------------------------------------------------------- #
def calcular_huella(archivos: list[Path], data_dir: Path, settings: Settings) -> str:
    """SHA-256 del contenido, de los parámetros de chunking y del modelo de embeddings."""
    proveedor, modelo = resolver_modelo_embeddings(settings)
    h = hashlib.sha256()
    for parte in (VERSION_FRAGMENTADOR, settings.chunk_size, settings.chunk_overlap, proveedor, modelo):
        h.update(f"{parte}\0".encode())
    for archivo in archivos:
        h.update(archivo.relative_to(data_dir).as_posix().encode() + b"\0")
        h.update(archivo.read_bytes() + b"\0")
    return h.hexdigest()


def ingest(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
    force: bool = False,
) -> ResultadoIngesta:
    """Indexa ``settings.data_dir`` en ChromaDB, salvo que ya esté al día.

    ``embeddings`` se puede inyectar (por ejemplo, en los tests); si no, se crea
    con ``build_embeddings`` solo cuando de verdad hay que indexar.
    """
    settings = settings or Settings.from_env()
    archivos = listar_archivos(settings.data_dir)
    huella = calcular_huella(archivos, settings.data_dir, settings)
    nombre = settings.collection_name

    cliente = abrir_cliente(settings.persist_dir)
    existente = obtener_coleccion(cliente, nombre)

    if existente is not None and not force:
        metadata = existente.metadata or {}
        if metadata.get(META_HUELLA) == huella and existente.count() > 0:
            logger.info(
                "La colección '%s' ya está al día (%d fragmentos); no se re-indexa.",
                nombre,
                existente.count(),
            )
            return ResultadoIngesta("sin_cambios", len(archivos), existente.count(), nombre)
        logger.info("Los documentos o la configuración cambiaron: se re-indexa '%s'.", nombre)

    documentos = cargar_documentos(archivos, settings.data_dir)
    fragmentos = fragmentar(documentos, settings)
    logger.info("%d archivo(s) -> %d fragmento(s)", len(documentos), len(fragmentos))

    embeddings = embeddings or build_embeddings(settings)
    proveedor, modelo = resolver_modelo_embeddings(settings)
    metadata_coleccion = {
        META_EMBEDDINGS_PROVIDER: proveedor,
        META_EMBEDDINGS_MODEL: modelo,
        META_HUELLA: huella,
        META_CHUNK_SIZE: settings.chunk_size,
        META_CHUNK_OVERLAP: settings.chunk_overlap,
        META_ARCHIVOS: len(documentos),
    }

    # Se indexa en una colección temporal y recién al final se reemplaza la
    # anterior: si falla la API de embeddings a mitad de camino, la base
    # existente queda intacta.
    temporal = nombre + "_tmp"
    if obtener_coleccion(cliente, temporal) is not None:
        cliente.delete_collection(temporal)
    vectorstore = Chroma(
        client=cliente,
        collection_name=temporal,
        embedding_function=embeddings,
        collection_metadata=metadata_coleccion,
        collection_configuration=CONFIGURACION_COLECCION,
    )
    try:
        vectorstore.add_documents(fragmentos, ids=[id_fragmento(f) for f in fragmentos])
    except Exception:
        cliente.delete_collection(temporal)
        raise

    if existente is not None:
        cliente.delete_collection(nombre)
    cliente.get_collection(temporal).modify(name=nombre)
    logger.info("Colección '%s' lista en %s", nombre, settings.persist_dir)
    return ResultadoIngesta("indexado", len(documentos), len(fragmentos), nombre)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Indexa ./data en ChromaDB.")
    parser.add_argument("--force", action="store_true", help="re-indexa aunque nada haya cambiado")
    parser.add_argument("--data-dir", type=Path, help="carpeta de documentos (default: data)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s | %(message)s")
    for ruidoso in ("httpx", "httpx2", "httpcore", "openai", "chromadb"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    try:
        settings = Settings.from_env(data_dir=args.data_dir)
        resultado = ingest(settings, force=args.force)
    except (ConfiguracionInvalidaError, SinDocumentosError, FaltaApiKeyError) as exc:
        logger.error("%s", exc)
        return 1

    print(
        f"\n{resultado.estado.upper()}: {resultado.archivos} archivo(s), "
        f"{resultado.fragmentos} fragmento(s) en la colección '{resultado.coleccion}'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
