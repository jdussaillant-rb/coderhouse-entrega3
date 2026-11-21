import pytest
from pydantic import ValidationError

from schemas import RESPUESTA_NO_SE, RespuestaLLM


def test_respuesta_valida():
    r = RespuestaLLM(respuesta=" 1,5 días ", encontrada_en_contexto=True, fragmentos_citados=["F2"])
    assert r.respuesta == "1,5 días"
    assert r.fragmentos_citados == ["F2"]


def test_normaliza_y_deduplica_ids():
    r = RespuestaLLM(
        respuesta="x", encontrada_en_contexto=True, fragmentos_citados=["[f1]", " F1 ", "F3"]
    )
    assert r.fragmentos_citados == ["F1", "F3"]


@pytest.mark.parametrize("id_invalido", ["fuente.md", "1", "F", "Fx"])
def test_rechaza_ids_invalidos(id_invalido):
    with pytest.raises(ValidationError):
        RespuestaLLM(respuesta="x", encontrada_en_contexto=True, fragmentos_citados=[id_invalido])


def test_respuesta_encontrada_sin_citas_se_rechaza():
    with pytest.raises(ValidationError, match="fragmentos_citados"):
        RespuestaLLM(respuesta="Son 15 días.", encontrada_en_contexto=True, fragmentos_citados=[])


def test_no_encontrada_se_normaliza_a_no_lo_se():
    r = RespuestaLLM(
        respuesta="No encontré esa información, pero suelen ser 15 días.",
        encontrada_en_contexto=False,
        fragmentos_citados=["F1"],
    )
    assert r.respuesta == RESPUESTA_NO_SE
    assert r.fragmentos_citados == []


def test_respuesta_vacia_se_rechaza():
    with pytest.raises(ValidationError):
        RespuestaLLM(respuesta="  ", encontrada_en_contexto=False)
