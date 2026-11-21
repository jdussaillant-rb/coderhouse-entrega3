"""Pruebas end-to-end del sistema RAG (con el LLM y los embeddings reales).

1. Ejecuta la ingesta (se salta si ``./vectorstore`` ya está al día).
2. Lanza en paralelo (``asyncio.gather``) tres preguntas:
   - una cuya respuesta SÍ está en los documentos,
   - una "pregunta trampa" plausible para la empresa, pero que los documentos no
     responden (el modelo tiende a inventar un número típico),
   - una "pregunta trampa" de conocimiento general, que el modelo conoce pero
     que no está en los documentos.
3. Verifica que la primera tenga respuesta con referencias y que las trampas
   devuelvan exactamente "No lo sé.". El código de salida es 1 si alguna falla.

Uso:
    python prueba_rag.py
    python prueba_rag.py --pregunta "¿Quién aprueba un failover de región?"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass

from config import ConfiguracionInvalidaError, Settings
from ingest import SinDocumentosError, ingest
from providers import FaltaApiKeyError
from rag import GeneracionFallidaError, RAGPipeline
from retriever import EmbeddingsIncompatiblesError, VectorstoreNoEncontradoError
from schemas import RESPUESTA_NO_SE, RespuestaRAG


@dataclass(frozen=True)
class Prueba:
    etiqueta: str
    pregunta: str
    esta_en_documentos: bool | None  # None = pregunta libre, sin verificación


PRUEBAS = [
    Prueba(
        "en_documentos",
        "¿Cómo se compensa una semana de guardia primaria y en qué plazo hay que tomar los días?",
        True,
    ),
    Prueba(
        "trampa_empresa",
        "¿Cuántos días de vacaciones al año tienen los empleados de Nimbus Cloud?",
        False,
    ),
    Prueba(
        "trampa_general",
        "¿Cuál es la capital de Australia?",
        False,
    ),
]


def configurar_logs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)-14s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for ruidoso in ("httpx", "httpx2", "httpcore", "openai", "anthropic", "chromadb"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def verificar(prueba: Prueba, resultado: RespuestaRAG) -> tuple[bool, str]:
    """¿El sistema se comportó como se esperaba?"""
    if prueba.esta_en_documentos is None:
        return True, "pregunta libre (sin verificación)"
    if prueba.esta_en_documentos:
        ok = resultado.encontrada_en_contexto and bool(resultado.referencias)
        return ok, "respondió con referencias" if ok else "NO respondió con referencias"
    ok = resultado.respuesta == RESPUESTA_NO_SE and not resultado.encontrada_en_contexto
    return ok, "dijo 'No lo sé.' (no alucinó)" if ok else "ALUCINÓ: respondió algo que no está en los documentos"


async def ejecutar(pruebas: list[Prueba], settings: Settings) -> int:
    resultado_ingesta = ingest(settings)
    print(
        f"\nIngesta: {resultado_ingesta.estado} "
        f"({resultado_ingesta.archivos} archivos, {resultado_ingesta.fragmentos} fragmentos)\n"
    )

    pipeline = RAGPipeline(settings)
    resultados = await asyncio.gather(
        *(pipeline.get_rag_response(p.pregunta) for p in pruebas), return_exceptions=True
    )

    fallas = 0
    for prueba, resultado in zip(pruebas, resultados):
        print(f"\n### {prueba.etiqueta}")
        if isinstance(resultado, BaseException):
            fallas += 1
            print(f"ERROR -> {type(resultado).__name__}: {resultado}")
            continue
        print(resultado.model_dump_json(indent=2))
        ok, detalle = verificar(prueba, resultado)
        fallas += not ok
        print(f"{'✔' if ok else '✘'} {detalle}")

    print(f"\nResumen: {len(pruebas) - fallas}/{len(pruebas)} pruebas OK")
    return 0 if fallas == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pruebas end-to-end del sistema RAG.")
    parser.add_argument(
        "--pregunta",
        action="append",
        help="pregunta libre (se puede repetir); reemplaza las pruebas predefinidas",
    )
    args = parser.parse_args(argv)
    configurar_logs()

    pruebas = (
        [Prueba(f"libre_{i}", p, None) for i, p in enumerate(args.pregunta, start=1)]
        if args.pregunta
        else PRUEBAS
    )
    try:
        return asyncio.run(ejecutar(pruebas, Settings.from_env()))
    except (
        ConfiguracionInvalidaError,
        SinDocumentosError,
        FaltaApiKeyError,
        VectorstoreNoEncontradoError,
        EmbeddingsIncompatiblesError,
        GeneracionFallidaError,
    ) as exc:
        logging.getLogger("rag").error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
