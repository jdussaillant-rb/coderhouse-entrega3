from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import providers
from config import Settings
from providers import Proveedor
from tests.fakes import EmbeddingsPalabras

DATA_REAL = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def embeddings() -> EmbeddingsPalabras:
    return EmbeddingsPalabras()


@pytest.fixture
def proveedor_fake(monkeypatch: pytest.MonkeyPatch, embeddings: EmbeddingsPalabras) -> EmbeddingsPalabras:
    """Registra el proveedor de embeddings 'fake' (sin API key) en providers.py."""
    monkeypatch.setitem(
        providers.PROVEEDORES_EMBEDDINGS,
        "fake",
        Proveedor(lambda modelo, settings: embeddings, "palabras-v1", None),
    )
    return embeddings


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Copia del dataset real en una carpeta temporal (los tests pueden modificarla)."""
    destino = tmp_path / "data"
    shutil.copytree(DATA_REAL, destino)
    return destino


@pytest.fixture
def settings(tmp_path: Path, data_dir: Path, proveedor_fake: EmbeddingsPalabras) -> Settings:
    return Settings(
        data_dir=data_dir,
        persist_dir=tmp_path / "vectorstore",
        collection_name="test_docs",
        embeddings_provider="fake",
    )
