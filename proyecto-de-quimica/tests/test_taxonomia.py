"""Pruebas de la Taxonomía Maestra (Etapa A) y de canecas.yaml (Etapa B)."""

import pytest

from taxonomia import Taxonomia, normalizar
from tests.test_mapeo import CATEGORIAS_TACO


@pytest.fixture(scope="module")
def tax():
    return Taxonomia.cargar()


def test_16_clases_con_ids_contiguos(tax):
    assert [c.id for c in tax.clases] == list(range(16))
    assert tax.nombres[15] == "fondo_sin_objeto"


def test_todas_las_clases_tienen_regla(tax):
    assert set(tax.reglas) == set(tax.nombres)


def test_las_60_categorias_de_taco(tax):
    estados = {c: tax.resolver("taco", c).estado for c in CATEGORIAS_TACO}
    assert [c for c, e in estados.items() if e == "sin_mapeo"] == []
    assert [c for c, e in estados.items() if e == "ambigua"] == ["Unlabeled litter"]


@pytest.mark.parametrize("fuente, etiqueta, clase", [
    ("taco", "Clear plastic bottle", "pet_botella"),
    ("taco", "Single-use carrier bag", "plastico_flexible_bolsa"),
    ("taco", "Tissues", "sanitario_y_rechazo"),
    ("taco", "Crisp packet", "multicapa_flexible"),
    ("taco", "Styrofoam piece", "icopor"),
    ("taco", "Drink carton", "multicapa_rigido"),
    ("zerowaste", "soft_plastic", "plastico_flexible_bolsa"),
    ("trashnet", "cardboard", "carton"),
    ("propias", "Bolsa blanca", "plastico_flexible_bolsa"),
    ("propias", "servilleta", "sanitario_y_rechazo"),
])
def test_alias(tax, fuente, etiqueta, clase):
    r = tax.resolver(fuente, etiqueta)
    assert r.estado == "mapeada" and r.clase == clase


def test_bolsa_y_servilleta_ya_no_comparten_clase(tax):
    """La causa de «bolsa blanca = servilleta»: antes ambas eran `trash`."""
    assert tax.resolver("taco", "Plastic film").clase != tax.resolver("taco", "Tissues").clase


def test_trashnet_plastic_es_ambigua(tax):
    r = tax.resolver("trashnet", "plastic")
    assert r.estado == "ambigua" and "pet_botella" in r.candidatas and not r.usable_clasificador


@pytest.mark.parametrize("packaging, categorias, clase", [
    (["en:bottle", "en:pet-1-polyethylene-terephthalate"], [], "pet_botella"),
    (["en:bag", "en:plastic"], ["en:snacks", "en:salty-snacks"], "multicapa_flexible"),
    (["en:tetra-pak"], ["en:milks"], "multicapa_rigido"),
    (["en:can"], [], "metal_lata"),
])
def test_reglas_open_food_facts(tax, packaging, categorias, clase):
    assert tax.resolver_off(packaging, categorias).clase == clase


def test_canecas_y_suciedad(tax):
    assert tax.caneca("plastico_flexible_bolsa") == "Blanca"
    assert tax.caneca("plastico_flexible_bolsa", sucio=True) == "Negra"
    assert tax.caneca("carton", sucio=True) == "Negra"
    assert tax.caneca("organico") == "Verde"
    assert tax.caneca("peligroso") == "Ninguna"
    assert tax.caneca("fondo_sin_objeto") is None


def test_perfil_operador(tax):
    t = Taxonomia.cargar(perfil="operador_recibe_multicapa")
    assert tax.caneca("multicapa_flexible") == "Negra"
    assert t.caneca("multicapa_flexible") == "Blanca"
    with pytest.raises(ValueError):
        Taxonomia.cargar(perfil="no_existe")


def test_decidir_incertidumbre(tax):
    assert tax.decidir({"papel": 0.52, "sanitario_y_rechazo": 0.41}).incierto
    d = tax.decidir({"plastico_flexible_bolsa": 0.9, "papel": 0.05})
    assert not d.incierto and d.caneca == "Blanca" and d.preguntar_limpieza


def test_normalizar():
    assert normalizar("en:pet-1-polyethylene-terephthalate") == "pet_1_polyethylene_terephthalate"
    assert normalizar("Rope & strings") == "rope_strings"
