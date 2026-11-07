"""
test_mapeo.py — Pruebas de mapeo de categorías críticas y normalización de nombres.

No descargan datasets (usa imágenes sintéticas y rutas ficticias).
Se ejecutan con RECICLAJE_DB temporal para no tocar la base real.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["RECICLAJE_DB"] = str(Path(__file__).parent.parent / "temp_pruebas.db")

from entrenamiento.reentrenar import (
    ClaseNormativa, _mapear_cat, MAPEO_TRASHNET_A_CLASE, MAPEO_TACO
)


def test_clase_normativa_crea_instancia():
    """La clase existe y se instancia."""
    cn = ClaseNormativa("Plástico")
    assert cn.nombre == "Plástico"
    print("✅ test_clase_normativa_crea_instancia")


def test_clase_normativa_mapeo_trashnet():
    """Las clases de TrashNet mapean correctamente."""
    assert MAPEO_TRASHNET_A_CLASE["plastic"].nombre == "Plástico"
    assert MAPEO_TRASHNET_A_CLASE["glass"].nombre == "Vidrio"
    assert MAPEO_TRASHNET_A_CLASE["metal"].nombre == "Metal"
    print("✅ test_clase_normativa_mapeo_trashnet")


def test_clase_normativa_mapeo_taco():
    """Las clases de TACO mapean correctamente."""
    assert MAPEO_TACO["Crisp packet"].nombre == "No aprovechable (Multicapa)"
    assert MAPEO_TACO["Aerosol"].nombre == "Metal (Blanco)"
    # Plastic bottle no está en TACO, usar otra categoría existente
    assert MAPEO_TACO["Glass bottle"].nombre == "Vidrio"
    print("✅ test_clase_normativa_mapeo_taco")


def test_mapear_cat_trashnet():
    """_mapear_cat funciona con nombres TrashNet."""
    cn = _mapear_cat("plastic")
    assert cn.nombre == "Plástico"
    cn = _mapear_cat("glass")
    assert cn.nombre == "Vidrio"
    print("✅ test_mapear_cat_trashnet")


def test_mapear_cat_taco():
    """_mapear_cat funciona con nombres TACO."""
    cn = _mapear_cat("Crisp packet")
    assert cn.nombre == "No aprovechable (Multicapa)"
    cn = _mapear_cat("Aerosol")
    assert cn.nombre == "Metal (Blanco)"
    print("✅ test_mapear_cat_taco")


def test_mapear_cat_faltante():
    """Categorías sin mapeo devuelven ClaseNormativa con evaluable=False."""
    cn = _mapear_cat("Food waste")
    assert cn.nombre == "Orgánico"
    assert cn.evaluable is False
    print("✅ test_mapear_cat_faltante")


def test_mapear_cat_ignore_case():
    """El mapeo ignora mayúsculas/minúsculas."""
    cn1 = _mapear_cat("CRISP PACKET")
    cn2 = _mapear_cat("crisp packet")
    assert cn1.nombre == cn2.nombre
    print("✅ test_mapear_cat_ignore_case")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
