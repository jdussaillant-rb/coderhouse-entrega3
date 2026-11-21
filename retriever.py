"""Capa de recuperación: pregunta -> embedding -> fragmentos más relevantes.

``abrir_vectorstore`` abre la colección creada por ``ingest.py`` y verifica que
el modelo de embeddings configurado sea el mismo con el que se indexó (el
"error #1" de la consigna: vectores de modelos distintos no son comparables).

``RetrieverConRelevancia`` es un ``BaseRetriever`` de LangChain (o sea, un
``Runnable[str, list[Document]]``) que hace la búsqueda de similitud con
``top_k`` fragmentos y deja la relevancia de cada uno en su metadata.
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from config import MAX_TOP_K, MIN_TOP_K, Settings
from providers import build_embeddings, resolver_modelo_embeddings
from store import META_EMBEDDINGS_MODEL, META_EMBEDDINGS_PROVIDER, abrir_cliente, obtener_coleccion

logger = logging.getLogger("rag.retriever")


def etiqueta_de(pregunta: str, largo: int = 30) -> str:
    """Comienzo de la pregunta, para distinguir en los logs consultas concurrentes."""
    plana = " ".join(pregunta.split())
    return plana if len(plana) <= largo else plana[: largo - 1] + "…"


class VectorstoreNoEncontradoError(RuntimeError):
    """No existe la colección (o está vacía): hay que correr ``python ingest.py``."""


class EmbeddingsIncompatiblesError(RuntimeError):
    """La colección se indexó con otro modelo de embeddings."""


def abrir_vectorstore(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    """Abre la colección persistente, validando que exista y que los embeddings coincidan."""
    if not settings.persist_dir.is_dir():
        raise VectorstoreNoEncontradoError(
            f"No existe {settings.persist_dir}. Ejecuta primero: python ingest.py"
        )
    cliente = abrir_cliente(settings.persist_dir)
    coleccion = obtener_coleccion(cliente, settings.collection_name)
    if coleccion is None or coleccion.count() == 0:
        raise VectorstoreNoEncontradoError(
            f"La colección '{settings.collection_name}' no existe o está vacía. "
            "Ejecuta primero: python ingest.py"
        )

    metadata = coleccion.metadata or {}
    indexado = (metadata.get(META_EMBEDDINGS_PROVIDER), metadata.get(META_EMBEDDINGS_MODEL))
    configurado = resolver_modelo_embeddings(settings)
    if indexado != configurado:
        raise EmbeddingsIncompatiblesError(
            f"La colección se indexó con {indexado[0]}/{indexado[1]} pero la consulta usaría "
            f"{configurado[0]}/{configurado[1]}. Usa el mismo modelo o re-indexa con "
            "python ingest.py --force"
        )

    return Chroma(
        client=cliente,
        collection_name=settings.collection_name,
        embedding_function=embeddings or build_embeddings(settings),
        create_collection_if_not_exists=False,
    )


class RetrieverConRelevancia(BaseRetriever):
    """Búsqueda de similitud en Chroma que conserva la relevancia de cada fragmento."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Chroma
    k: int = Field(default=4, ge=MIN_TOP_K, le=MAX_TOP_K)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        resultados = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.k)
        return self._con_relevancia(query, resultados)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        resultados = await self.vectorstore.asimilarity_search_with_relevance_scores(
            query, k=self.k
        )
        return self._con_relevancia(query, resultados)

    @staticmethod
    def _con_relevancia(query: str, resultados: list[tuple[Document, float]]) -> list[Document]:
        documentos: list[Document] = []
        for documento, relevancia in resultados:
            documento.metadata["relevancia"] = round(float(relevancia), 4)
            documentos.append(documento)
        logger.info(
            "[%s] Recuperados %d fragmento(s): %s",
            etiqueta_de(query),
            len(documentos),
            ", ".join(
                f"{d.metadata.get('source')}#{d.metadata.get('chunk')} ({d.metadata['relevancia']:.2f})"
                for d in documentos
            ),
        )
        return documentos


def build_retriever(
    settings: Settings | None = None, embeddings: Embeddings | None = None
) -> RetrieverConRelevancia:
    """Retriever listo para usar, con ``top_k`` de la configuración."""
    settings = settings or Settings.from_env()
    return RetrieverConRelevancia(
        vectorstore=abrir_vectorstore(settings, embeddings), k=settings.top_k
    )
