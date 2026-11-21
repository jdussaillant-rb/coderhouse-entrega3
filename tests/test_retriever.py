import pytest

import providers
from config import Settings
from ingest import ingest
from providers import Proveedor
from retriever import (
    EmbeddingsIncompatiblesError,
    RetrieverConRelevancia,
    VectorstoreNoEncontradoError,
    abrir_vectorstore,
    build_retriever,
)
from tests.fakes import EmbeddingsPalabras

# Los embeddings falsos son por bolsa de palabras: la pregunta usa palabras del fragmento.
PREGUNTA_GUARDIA = "¿Cuántas horas libres suma cada incidente atendido fuera del horario laboral?"


def test_sin_vectorstore_pide_correr_la_ingesta(settings: Settings):
    with pytest.raises(VectorstoreNoEncontradoError, match="ingest.py"):
        build_retriever(settings)


def test_coleccion_inexistente(settings: Settings):
    ingest(settings)
    otra = settings.model_copy(update={"collection_name": "otra_coleccion"})
    with pytest.raises(VectorstoreNoEncontradoError):
        build_retriever(otra)


def test_embeddings_distintos_al_indexado_se_rechazan(settings: Settings):
    ingest(settings)
    distinto = settings.model_copy(update={"embeddings_model": "otro-modelo"})
    with pytest.raises(EmbeddingsIncompatiblesError, match="fake/palabras-v1"):
        abrir_vectorstore(distinto)


def test_proveedor_distinto_al_indexado_se_rechaza(settings: Settings, monkeypatch):
    ingest(settings)
    monkeypatch.setitem(
        providers.PROVEEDORES_EMBEDDINGS,
        "otro",
        Proveedor(lambda m, s: EmbeddingsPalabras(), "palabras-v1", None),
    )
    with pytest.raises(EmbeddingsIncompatiblesError):
        abrir_vectorstore(settings.model_copy(update={"embeddings_provider": "otro"}))


async def test_recupera_top_k_con_relevancia(settings: Settings):
    ingest(settings)
    retriever = build_retriever(settings)

    documentos = await retriever.ainvoke(PREGUNTA_GUARDIA)

    assert len(documentos) == settings.top_k
    assert documentos[0].metadata["source"] == "02_runbook_incidentes.md"
    assert "guardia" in documentos[0].page_content.lower()
    relevancias = [d.metadata["relevancia"] for d in documentos]
    assert relevancias == sorted(relevancias, reverse=True)
    assert all(-1.0 <= r <= 1.0 for r in relevancias)


def test_version_sincrona(settings: Settings):
    ingest(settings)
    documentos = build_retriever(settings).invoke(PREGUNTA_GUARDIA)
    assert len(documentos) == settings.top_k


def test_top_k_fuera_de_rango(settings: Settings):
    ingest(settings)
    vectorstore = abrir_vectorstore(settings)
    with pytest.raises(ValueError):
        RetrieverConRelevancia(vectorstore=vectorstore, k=10)
