"""Pruebas de database.py: validación por grupos y consultas del tablero."""

import pytest

import database as db


@pytest.mark.parametrize("material, grupo", [
    ("No aprovechable (Multicapa)", "No aprovechable"),
    ("No aprovechable (Icopor)", "No aprovechable"),
    ("Metal (Blanco)", "Metal"),
    ("Cartón (Tetra Pak)", "Papel/Cartón"),
    ("Papel (Blanco)", "Papel/Cartón"),
    ("Plástico (Blanco)", "Plástico"),
    ("PET", "Plástico"),
    ("PEAD", "Plástico"),
    ("Papel/Cartón", "Papel/Cartón"),
    ("No identificado", "No identificado"),
])
def test_grupo_material(material, grupo):
    assert db.grupo_material(material) == grupo


def _poblar():
    db.crear_lote([
        {"material": "Papel/Cartón", "confianza": 90, "contenedor": "Blanco", "origen": "taco",
         "motor": "m", "etiqueta_real": "No aprovechable (Icopor)", "archivo": "a#1",
         "fecha_hora": "2026-11-01 08:00:00"},
        {"material": "No aprovechable", "confianza": 80, "contenedor": "Negro", "origen": "taco",
         "motor": "m", "etiqueta_real": "No aprovechable (Multicapa)", "archivo": "a#2",
         "fecha_hora": "2026-11-02 09:00:00"},
        {"material": "Metal", "confianza": 70, "contenedor": "Blanco", "origen": "camara"},
    ])


def test_crear_lote_ignora_duplicados():
    fila = {"material": "Metal", "confianza": 70, "contenedor": "Blanco", "origen": "taco",
            "motor": "m", "archivo": "x#1"}
    assert db.crear_lote([fila]) == 1
    assert db.crear_lote([fila]) == 0


def test_validaciones():
    with pytest.raises(ValueError):
        db.crear_clasificacion("Metal", 70, "Azul")
    with pytest.raises(ValueError):
        db.crear_clasificacion("Metal", 150, "Blanco")


def test_funciones_del_tablero():
    _poblar()
    assert db.resumen_kpis()["total"] == 3
    assert set(db.origenes_disponibles()) == {"camara", "taco"}
    assert db.conteo_por_material()["cantidad"].sum() == 3
    assert db.proporcion_aprovechable()["cantidad"].sum() == 3
    assert not db.volumen_por_dia().empty
    assert db.listar_clasificaciones(origenes=["taco"]).shape[0] == 2


def test_validacion_del_tablero_compara_por_grupo():
    """Misma lógica que app.seccion_validacion: icopor->papel es error; multicapa->no aprov. acierta."""
    _poblar()
    df = db.datos_validacion()
    df["acierto"] = df["material"].map(db.grupo_material) == df["etiqueta_real"].map(db.grupo_material)
    resultado = dict(zip(df["etiqueta_real"], df["acierto"]))
    assert resultado == {"No aprovechable (Icopor)": False, "No aprovechable (Multicapa)": True}
