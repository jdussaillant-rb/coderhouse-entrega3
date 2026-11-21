from pathlib import Path

import pytest
import tiktoken

from config import Settings
from ingest import (
    ENCODING_TIKTOKEN,
    SinDocumentosError,
    cargar_documentos,
    fragmentar,
    id_fragmento,
    ingest,
    listar_archivos,
)
from store import META_EMBEDDINGS_MODEL, META_EMBEDDINGS_PROVIDER, META_HUELLA, abrir_cliente
from tests.fakes import EmbeddingsPalabras

ENC = tiktoken.get_encoding(ENCODING_TIKTOKEN)


def tokens(texto: str) -> int:
    return len(ENC.encode(texto))


def coleccion(settings: Settings):
    return abrir_cliente(settings.persist_dir).get_collection(settings.collection_name)


# --- Carga -------------------------------------------------------------------


def test_lista_solo_md_y_txt_recursivo_y_ordenado(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    for nombre in ("b.md", "a.txt", "sub/c.MD", "imagen.png", "notas.pdf"):
        (tmp_path / nombre).write_text("contenido", encoding="utf-8")
    archivos = [p.relative_to(tmp_path).as_posix() for p in listar_archivos(tmp_path)]
    assert archivos == ["a.txt", "b.md", "sub/c.MD"]


def test_carpeta_inexistente_o_sin_documentos(tmp_path: Path):
    with pytest.raises(SinDocumentosError):
        listar_archivos(tmp_path / "no-existe")
    (tmp_path / "foto.png").write_bytes(b"x")
    with pytest.raises(SinDocumentosError):
        listar_archivos(tmp_path)


def test_omite_archivos_vacios(tmp_path: Path):
    (tmp_path / "vacio.md").write_text("   \n", encoding="utf-8")
    (tmp_path / "lleno.md").write_text("# Hola\nTexto", encoding="utf-8")
    documentos = cargar_documentos(listar_archivos(tmp_path), tmp_path)
    assert [d.metadata["source"] for d in documentos] == ["lleno.md"]

    (tmp_path / "lleno.md").write_text("", encoding="utf-8")
    with pytest.raises(SinDocumentosError):
        cargar_documentos(listar_archivos(tmp_path), tmp_path)


# --- Chunking ----------------------------------------------------------------


@pytest.fixture
def fragmentos(data_dir: Path):
    documentos = cargar_documentos(listar_archivos(data_dir), data_dir)
    return documentos, fragmentar(documentos, Settings())


def test_fragmentos_respetan_500_tokens(fragmentos):
    _, frags = fragmentos
    assert len(frags) > 4  # cada archivo se dividió en varios fragmentos
    assert all(tokens(f.page_content) <= 500 for f in frags)


def test_fragmentos_consecutivos_se_solapan_hasta_50_tokens(fragmentos):
    documentos, frags = fragmentos
    textos = {d.metadata["source"]: d.page_content for d in documentos}
    solapes = []
    for a, b in zip(frags, frags[1:]):
        if a.metadata["source"] != b.metadata["source"]:
            continue
        fin_a = a.metadata["start_index"] + len(a.page_content)
        compartido = textos[a.metadata["source"]][b.metadata["start_index"] : fin_a]
        solapes.append(tokens(compartido) if fin_a > b.metadata["start_index"] else 0)
    assert solapes and all(s <= 50 for s in solapes)
    assert all(s > 0 for s in solapes), f"hay fragmentos consecutivos sin solapamiento: {solapes}"


def test_metadata_de_cada_fragmento(fragmentos):
    documentos, frags = fragmentos
    textos = {d.metadata["source"]: d.page_content for d in documentos}
    for f in frags:
        inicio = f.metadata["start_index"]
        assert inicio >= 0
        assert textos[f.metadata["source"]][inicio : inicio + len(f.page_content)] == f.page_content
        assert isinstance(f.metadata["chunk"], int)
        assert f.metadata["section"]
    por_archivo = [f.metadata["chunk"] for f in frags if f.metadata["source"] == "02_runbook_incidentes.md"]
    assert por_archivo == list(range(len(por_archivo)))


def test_seccion_lista_las_secciones_que_abarca(fragmentos):
    _, frags = fragmentos
    guardia = next(f for f in frags if "Compensación: cada semana de guardia" in f.page_content)
    assert guardia.metadata["source"] == "02_runbook_incidentes.md"
    assert "3. Guardia (on-call)" in guardia.metadata["section"].split(" › ")
    # El título del documento (nivel 1) no aparece si hay secciones más específicas.
    assert all("Runbook de gestión" not in f.metadata["section"] for f in frags if f.metadata["chunk"] > 0)


def test_ids_deterministas(fragmentos):
    _, frags = fragmentos
    ids = [id_fragmento(f) for f in frags]
    assert len(ids) == len(set(ids))
    assert ids[0] == "01_arquitectura_plataforma.md::0"


# --- Persistencia ------------------------------------------------------------


def test_ingesta_crea_coleccion_con_metadata(settings: Settings, embeddings: EmbeddingsPalabras):
    resultado = ingest(settings)

    assert resultado.estado == "indexado"
    assert resultado.archivos == 4
    col = coleccion(settings)
    assert col.count() == resultado.fragmentos == embeddings.textos_embebidos
    assert col.metadata[META_EMBEDDINGS_PROVIDER] == "fake"
    assert col.metadata[META_EMBEDDINGS_MODEL] == "palabras-v1"
    assert len(col.metadata[META_HUELLA]) == 64


def test_no_reindexa_si_nada_cambio(settings: Settings, embeddings: EmbeddingsPalabras):
    ingest(settings)
    llamadas = embeddings.llamadas

    resultado = ingest(settings)

    assert resultado.estado == "sin_cambios"
    assert embeddings.llamadas == llamadas  # no se gastaron embeddings


def test_reindexa_si_cambia_un_documento(settings: Settings, data_dir: Path):
    primero = ingest(settings)
    (data_dir / "05_nuevo.txt").write_text(
        "Política de mascotas: se permiten perros en la oficina los viernes. " * 40,
        encoding="utf-8",
    )

    segundo = ingest(settings)

    assert segundo.estado == "indexado"
    assert segundo.archivos == 5
    assert coleccion(settings).count() == segundo.fragmentos > primero.fragmentos


def test_reindexa_si_cambia_el_chunking(settings: Settings):
    ingest(settings)
    otra = settings.model_copy(update={"chunk_size": 800, "chunk_overlap": 80})
    assert ingest(otra).estado == "indexado"


def test_force_reindexa_sin_duplicar(settings: Settings):
    primero = ingest(settings)
    segundo = ingest(settings, force=True)
    assert segundo.estado == "indexado"
    assert coleccion(settings).count() == primero.fragmentos


def test_si_fallan_los_embeddings_la_coleccion_anterior_queda_intacta(settings: Settings, data_dir: Path):
    primero = ingest(settings)
    (data_dir / "01_arquitectura_plataforma.md").write_text("# Cambio\nTexto nuevo", encoding="utf-8")

    with pytest.raises(ConnectionError):
        ingest(settings, embeddings=EmbeddingsPalabras(falla_en_llamada=1))

    cliente = abrir_cliente(settings.persist_dir)
    assert cliente.get_collection(settings.collection_name).count() == primero.fragmentos
    assert [c.name for c in cliente.list_collections()] == [settings.collection_name]
