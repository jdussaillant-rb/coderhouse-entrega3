import pytest

import config
from config import ConfiguracionInvalidaError, Settings


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aísla los tests del .env local y de variables del sistema."""
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    for variable in config.VARIABLES_DE_ENTORNO:
        monkeypatch.delenv(variable, raising=False)


def test_defaults_cumplen_la_consigna():
    s = Settings()
    assert s.chunk_size == 500
    assert s.chunk_overlap == 50
    assert 3 <= s.top_k <= 5
    assert s.embeddings_provider == "openai"


@pytest.mark.parametrize("top_k", [3, 4, 5])
def test_top_k_valido(top_k):
    assert Settings(top_k=top_k).top_k == top_k


@pytest.mark.parametrize("top_k", [1, 2, 6, 50])
def test_top_k_fuera_de_rango_se_rechaza(top_k):
    with pytest.raises(ValueError):
        Settings(top_k=top_k)


def test_chunk_menor_a_500_tokens_se_rechaza():
    with pytest.raises(ValueError):
        Settings(chunk_size=400)


def test_solapamiento_menor_a_50_se_rechaza():
    with pytest.raises(ValueError):
        Settings(chunk_overlap=40)


def test_solapamiento_no_puede_superar_el_fragmento():
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_size=500, chunk_overlap=500)


def test_es_inmutable():
    s = Settings()
    with pytest.raises(ValueError):
        s.top_k = 5


def test_from_env_lee_variables(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "5")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "800")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    s = Settings.from_env()
    assert (s.top_k, s.chunk_size, s.llm_provider) == (5, 800, "anthropic")


def test_from_env_ignora_variables_vacias(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("RAG_TOP_K", "  ")
    s = Settings.from_env()
    assert s.llm_model is None
    assert s.top_k == 4


def test_from_env_error_nombra_la_variable(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "20")
    with pytest.raises(ConfiguracionInvalidaError, match="RAG_TOP_K"):
        Settings.from_env()


def test_overrides_tienen_prioridad(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "5")
    assert Settings.from_env(top_k=3).top_k == 3
