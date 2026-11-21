import asyncio
import re

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage

from config import Settings
from ingest import ingest
from rag import (
    MAX_CARACTERES_PREGUNTA,
    MAX_INTENTOS,
    GeneracionFallidaError,
    RAGPipeline,
    build_prompt,
    formatear_contexto,
)
from schemas import RESPUESTA_NO_SE, RespuestaLLM, RespuestaRAG
from tests.fakes import (
    ModeloGuionado,
    RetrieverFijo,
    mensaje_texto,
    respuesta_json,
    respuesta_no_se,
)

DOCUMENTOS = [
    Document(
        page_content="Cada semana de guardia primaria se compensa con 1,5 días libres.",
        metadata={"source": "02_runbook_incidentes.md", "chunk": 1, "section": "3. Guardia (on-call)", "relevancia": 0.81},
    ),
    Document(
        page_content="Los despliegues se permiten de lunes a jueves.",
        metadata={"source": "03_politica_despliegues.md", "chunk": 2, "section": "5. Ventanas de despliegue", "relevancia": 0.42},
    ),
    Document(
        page_content="El periodo de prueba dura 3 meses.",
        metadata={"source": "04_guia_onboarding.md", "chunk": 3, "section": "8. Periodo de prueba", "relevancia": 0.30},
    ),
]
PREGUNTA = "¿Cómo se compensa la guardia?"


def pipeline_con(*respuestas: AIMessage, retriever: RetrieverFijo | None = None):
    modelo = ModeloGuionado(respuestas=list(respuestas))
    retriever = retriever or RetrieverFijo(documentos=DOCUMENTOS)
    pipeline = RAGPipeline(Settings(), retriever=retriever, llm=modelo, esperar_entre_intentos=False)
    return modelo, retriever, pipeline


# --- Prompt y transformador de documentos -----------------------------------


def test_prompt_tiene_filtro_de_veracidad_e_instrucciones_de_formato():
    prompt = build_prompt()
    assert sorted(prompt.input_variables) == ["contexto", "pregunta"]

    sistema, humano = prompt.format_messages(contexto="CTX {con llaves}", pregunta="P?")
    assert "No lo sé." in sistema.content
    assert "EXCLUSIVAMENTE" in sistema.content
    for campo in RespuestaLLM.model_fields:
        assert campo in sistema.content
    assert "CTX {con llaves}" in humano.content
    assert "<pregunta>\nP?\n</pregunta>" in humano.content


def test_formatear_contexto_numera_fragmentos_con_su_fuente():
    contexto = formatear_contexto(DOCUMENTOS)
    assert contexto.startswith("[F1] fuente: 02_runbook_incidentes.md | sección: 3. Guardia (on-call)")
    assert "[F2] fuente: 03_politica_despliegues.md" in contexto
    assert "[F3]" in contexto and "[F4]" not in contexto
    assert formatear_contexto([]) == "(sin fragmentos)"


# --- Respuestas ----------------------------------------------------------------


async def test_respuesta_con_referencias_verificadas():
    modelo, retriever, pipeline = pipeline_con(respuesta_json(citados=["F1"]))

    r = await pipeline.get_rag_response(PREGUNTA)

    assert isinstance(r, RespuestaRAG)
    assert r.pregunta == PREGUNTA
    assert r.encontrada_en_contexto is True
    assert r.respuesta == "Se compensa con 1,5 días libres."
    [ref] = r.referencias
    assert (ref.id, ref.fuente, ref.seccion, ref.fragmento, ref.relevancia) == (
        "F1", "02_runbook_incidentes.md", "3. Guardia (on-call)", 1, 0.81,
    )
    assert ref.extracto.startswith("Cada semana de guardia")
    assert modelo.llamadas == 1 and retriever.llamadas == 1


async def test_el_llm_recibe_los_fragmentos_recuperados():
    modelo, _, pipeline = pipeline_con(respuesta_json())
    await pipeline.get_rag_response(PREGUNTA)

    humano = modelo.prompts_recibidos[0][-1].content
    assert "[F1] fuente: 02_runbook_incidentes.md" in humano
    assert "1,5 días libres" in humano
    assert PREGUNTA in humano


async def test_acepta_json_dentro_de_bloque_markdown():
    _, _, pipeline = pipeline_con(respuesta_json(con_bloque_markdown=True))
    r = await pipeline.get_rag_response(PREGUNTA)
    assert r.encontrada_en_contexto


async def test_pregunta_trampa_devuelve_no_lo_se():
    _, _, pipeline = pipeline_con(respuesta_no_se())

    r = await pipeline.get_rag_response("¿Cuál es la capital de Australia?")

    assert r.respuesta == RESPUESTA_NO_SE
    assert r.encontrada_en_contexto is False
    assert r.referencias == []


async def test_descarta_citas_a_fragmentos_no_recuperados():
    _, _, pipeline = pipeline_con(respuesta_json(citados=["F1", "F9"]))
    r = await pipeline.get_rag_response(PREGUNTA)
    assert [ref.id for ref in r.referencias] == ["F1"]


async def test_si_ninguna_cita_es_valida_responde_no_lo_se():
    _, _, pipeline = pipeline_con(respuesta_json("Son 15 días.", citados=["F7"]))
    r = await pipeline.get_rag_response(PREGUNTA)
    assert r.respuesta == RESPUESTA_NO_SE
    assert r.encontrada_en_contexto is False


# --- Reintentos ----------------------------------------------------------------


@pytest.mark.parametrize(
    "primera",
    [
        mensaje_texto('{"respuesta": "Se compensa con 1,5 d'),  # JSON cortado
        mensaje_texto("Se compensa con 1,5 días libres."),  # texto libre, sin JSON
        respuesta_json(citados=[]),  # "encontrada" sin citas
        respuesta_json(finish_reason="length"),  # truncado por max_tokens (OpenAI)
        respuesta_json(finish_reason="max_tokens"),  # truncado (Anthropic)
    ],
    ids=["json_cortado", "texto_libre", "sin_citas", "truncado_openai", "truncado_anthropic"],
)
async def test_reintenta_ante_salida_invalida(primera, caplog):
    modelo, retriever, pipeline = pipeline_con(primera, respuesta_json())

    r = await pipeline.get_rag_response(PREGUNTA)

    assert r.encontrada_en_contexto
    assert modelo.llamadas == 2
    assert retriever.llamadas == 1  # la búsqueda no se repite
    assert "Reintento automático: intento 2" in caplog.text


async def test_reintentos_agotados():
    modelo, _, pipeline = pipeline_con(mensaje_texto("no es json"))

    with pytest.raises(GeneracionFallidaError, match=f"{MAX_INTENTOS} intento"):
        await pipeline.get_rag_response(PREGUNTA)
    assert modelo.llamadas == MAX_INTENTOS


async def test_negativa_del_modelo_no_se_reintenta():
    modelo, _, pipeline = pipeline_con(respuesta_json(finish_reason="refusal"))
    with pytest.raises(GeneracionFallidaError, match="negó"):
        await pipeline.get_rag_response(PREGUNTA)
    assert modelo.llamadas == 1


# --- Validación de la entrada --------------------------------------------------


@pytest.mark.parametrize("pregunta", ["", "   ", None, "x" * (MAX_CARACTERES_PREGUNTA + 1)])
async def test_pregunta_invalida(pregunta):
    modelo, retriever, pipeline = pipeline_con(respuesta_json())
    with pytest.raises(ValueError):
        await pipeline.get_rag_response(pregunta)
    assert modelo.llamadas == 0 and retriever.llamadas == 0


# --- Concurrencia -------------------------------------------------------------


async def test_preguntas_concurrentes_no_se_mezclan():
    def responder(mensajes: list[BaseMessage]) -> AIMessage:
        pregunta = re.search(r"<pregunta>\n(.*)\n</pregunta>", mensajes[-1].content).group(1)
        return respuesta_json(f"eco: {pregunta}")

    modelo = ModeloGuionado(responder=responder)
    pipeline = RAGPipeline(
        Settings(), retriever=RetrieverFijo(documentos=DOCUMENTOS), llm=modelo, esperar_entre_intentos=False
    )
    preguntas = [f"pregunta {i}" for i in range(5)]

    resultados = await asyncio.gather(*(pipeline.get_rag_response(p) for p in preguntas))

    assert [r.respuesta for r in resultados] == [f"eco: {p}" for p in preguntas]
    assert [r.pregunta for r in resultados] == preguntas


# --- End-to-end: ingesta real + ChromaDB + cadena ------------------------------


async def test_end_to_end_con_chroma(settings: Settings):
    ingest(settings)
    modelo = ModeloGuionado(respuestas=[respuesta_json("1,5 días libres por semana.", citados=["F1"])])
    pipeline = RAGPipeline(settings, llm=modelo, esperar_entre_intentos=False)

    r = await pipeline.get_rag_response("¿Cuántas horas libres suma cada incidente atendido fuera del horario laboral?")

    assert r.referencias[0].fuente == "02_runbook_incidentes.md"
    assert r.referencias[0].relevancia is not None
    contexto = modelo.prompts_recibidos[0][-1].content
    assert contexto.count("fuente:") == settings.top_k
