import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import Settings
from providers import (
    FaltaApiKeyError,
    ProveedorNoSoportadoError,
    build_embeddings,
    build_llm,
    resolver_modelo_embeddings,
)


def test_modelo_de_embeddings_por_defecto():
    assert resolver_modelo_embeddings(Settings()) == ("openai", "text-embedding-3-small")


def test_modelo_de_embeddings_configurable():
    s = Settings(embeddings_model="text-embedding-3-large")
    assert resolver_modelo_embeddings(s) == ("openai", "text-embedding-3-large")


def test_proveedor_desconocido_lista_las_opciones():
    with pytest.raises(ProveedorNoSoportadoError, match="anthropic, openai"):
        build_llm(Settings(llm_provider="gemini"))


def test_anthropic_no_ofrece_embeddings():
    with pytest.raises(ProveedorNoSoportadoError, match="EMBEDDINGS_PROVIDER"):
        build_embeddings(Settings(embeddings_provider="anthropic"))


def test_falta_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(FaltaApiKeyError, match="OPENAI_API_KEY"):
        build_llm(Settings())


def test_crea_clientes_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = build_llm(Settings(llm_max_tokens=512))
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"
    assert llm.max_tokens == 512
    emb = build_embeddings(Settings())
    assert isinstance(emb, OpenAIEmbeddings)
    assert emb.model == "text-embedding-3-small"


def test_crea_cliente_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    llm = build_llm(Settings(llm_provider="anthropic"))
    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "claude-haiku-5-5"
