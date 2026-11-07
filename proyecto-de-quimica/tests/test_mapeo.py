"""Pruebas del mapeo de categorías de datasets a clases normativas (Res. 2184 de 2019)."""

import pytest

import database as db
import ingesta_publica as ip

# Las 60 categorías oficiales de TACO (data/annotations.json).
CATEGORIAS_TACO = [
    "Aluminium foil", "Battery", "Aluminium blister pack", "Carded blister pack",
    "Other plastic bottle", "Clear plastic bottle", "Glass bottle", "Plastic bottle cap",
    "Metal bottle cap", "Broken glass", "Food Can", "Aerosol", "Drink can", "Toilet tube",
    "Other carton", "Egg carton", "Drink carton", "Corrugated carton", "Meal carton",
    "Pizza box", "Paper cup", "Disposable plastic cup", "Foam cup", "Glass cup",
    "Other plastic cup", "Food waste", "Glass jar", "Plastic lid", "Metal lid",
    "Other plastic", "Magazine paper", "Tissues", "Wrapping paper", "Normal paper",
    "Paper bag", "Plastified paper bag", "Plastic film", "Six pack rings", "Garbage bag",
    "Other plastic wrapper", "Single-use carrier bag", "Polypropylene bag", "Crisp packet",
    "Spread tub", "Tupperware", "Disposable food container", "Foam food container",
    "Other plastic container", "Plastic glooves", "Plastic utensils", "Pop tab",
    "Rope & strings", "Scrap metal", "Shoe", "Squeezable tube", "Plastic straw",
    "Paper straw", "Styrofoam piece", "Unlabeled litter", "Cigarette",
]


@pytest.mark.parametrize("categoria, esperado, contenedor", [
    ("Crisp packet", "No aprovechable (Multicapa)", "Negro"),
    ("crisp_packet", "No aprovechable (Multicapa)", "Negro"),
    ("Foam food container", "No aprovechable (Icopor)", "Negro"),
    ("foam_food_container", "No aprovechable (Icopor)", "Negro"),
    ("Styrofoam piece", "No aprovechable (Icopor)", "Negro"),
    ("Aerosol", "Metal (Blanco)", "Blanco"),
    ("icopor", "No aprovechable (Icopor)", "Negro"),
    ("paquete_papas", "No aprovechable (Multicapa)", "Negro"),
    ("Drink carton", "Cartón (Tetra Pak)", "Blanco"),
])
def test_clases_criticas(categoria, esperado, contenedor):
    clase = ip.mapear_categoria(categoria)
    assert clase is not None
    assert clase.nombre == esperado
    assert clase.contenedor == contenedor


def test_respaldo_por_supercategoria():
    """Una categoría desconocida se resuelve por su súper-categoría."""
    assert ip.mapear_categoria("Lata de atún", "Can").nombre == "Metal (Blanco)"
    assert ip.mapear_categoria("Cosa rara", "Categoria sin mapeo") is None


def test_normalizar_nombre():
    assert ip.normalizar_nombre("Foam food container") == "foam_food_container"
    assert ip.normalizar_nombre("Cartón  Ondulado") == "carton_ondulado"
    assert ip.normalizar_nombre("Rope & strings") == "rope_strings"


def test_las_60_categorias_de_taco_tienen_clase():
    sin_clase = [c for c in CATEGORIAS_TACO if ip.mapear_categoria(c) is None]
    assert len(CATEGORIAS_TACO) == 60
    assert sin_clase == []


def test_destinos_validos_y_grupos_con_tilde():
    """Todo destino existe y su grupo coincide EXACTAMENTE con los grupos del motor."""
    grupos_motor = {db.grupo_material(m) for m in
                    ("PET", "PEAD", "Vidrio", "Metal", "Papel/Cartón", "No aprovechable")}
    for nombre, clase in ip.CLASES_NORMATIVAS.items():
        assert nombre == clase.nombre
        if clase.evaluable:
            assert clase.grupo in grupos_motor, f"{nombre} -> {clase.grupo}"


def test_no_evaluables():
    for categoria in ("Food waste", "Battery", "Unlabeled litter"):
        assert ip.mapear_categoria(categoria).evaluable is False
