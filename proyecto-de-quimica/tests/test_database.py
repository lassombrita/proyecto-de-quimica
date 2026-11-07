"""
test_database.py — Pruebas de grupo_material y funciones del tablero.

Verifica la lógica de agrupación de materiales y las consultas analíticas.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["RECICLAJE_DB"] = str(Path(__file__).parent.parent / "temp_pruebas.db")

from database import grupo_material, resumen_kpis, conteo_por_material, \
    proporcion_aprovechable, volumen_por_dia, datos_validacion


def test_grupo_material_no_ape():
    """grupo_material reduce 'No aprovechable (Multicapa)' a 'No aprovechable'."""
    result = grupo_material("No aprovechable (Multicapa)")
    assert result == "No aprovechable", \
        f"Se esperaba 'No aprovechable', got {result!r}"
    
    print("✅ test_grupo_material_no_ape")


def test_grupo_material_metal():
    """grupo_material reduce 'Metal (Blanco)' a 'Metal'."""
    result = grupo_material("Metal (Blanco)")
    assert result == "Metal", \
        f"Se esperaba 'Metal', got {result!r}"
    
    print("✅ test_grupo_material_metal")


def test_grupo_material_pet():
    """grupo_material reduce 'PET' a 'Plástico'."""
    result = grupo_material("PET")
    assert result == "Plástico", \
        f"Se esperaba 'Plástico', got {result!r}"
    
    print("✅ test_grupo_material_pet")


def test_grupo_material_pead():
    """grupo_material reduce 'PEAD' a 'Plástico'."""
    result = grupo_material("PEAD")
    assert result == "Plástico", \
        f"Se esperaba 'Plástico', got {result!r}"
    
    print("✅ test_grupo_material_pead")


def test_grupo_material_vidrio():
    """grupo_material mantiene 'Vidrio' como grupo."""
    result = grupo_material("Vidrio")
    assert result == "Vidrio", \
        f"Se esperaba 'Vidrio', got {result!r}"
    
    print("✅ test_grupo_material_vidrio")


def test_grupo_material_papel_carton():
    """grupo_material mantiene 'Papel/Cartón' como grupo."""
    result = grupo_material("Papel/Cartón")
    assert result == "Papel/Cartón", \
        f"Se esperaba 'Papel/Cartón', got {result!r}"
    
    print("✅ test_grupo_material_papel_carton")


def test_resumen_kpis():
    """resumen_kpis devuelve diccionario con campos esperados."""
    result = resumen_kpis()
    assert isinstance(result, dict), "Se esperaba un diccionario"
    
    campos_esperados = ["total", "confianza_promedio", "aprovechables", 
                        "primera", "ultima"]
    for campo in campos_esperados:
        assert campo in result, f"Falta campo '{campo}' en resumen_kpis()"
    
    # total debe ser un número
    assert isinstance(result["total"], (int, float))
    
    print("✅ test_resumen_kpis")


def test_conteo_por_material():
    """conteo_por_material devuelve DataFrame con campos esperados."""
    df = conteo_por_material()
    assert df is not None, "Debe devolver un DataFrame"
    
    # Debería tener al menos estas columnas
    if not df.empty:
        cols_esperadas = ["material", "contenedor", "cantidad", 
                          "confianza_media"]
        for col in cols_esperadas:
            assert col in df.columns, \
                f"Debe tener columna '{col}', tiene {list(df.columns)}"
    
    print("✅ test_conteo_por_material")


def test_proporcion_aprovechable():
    """proporcion_aprovechable devuelve DataFrame agrupado por tipo."""
    df = proporcion_aprovechable()
    assert df is not None, "Debe devolver un DataFrame"
    
    # Debería tener al menos estas columnas
    if not df.empty:
        assert "tipo" in df.columns, \
            f"Debe tener columna 'tipo', tiene {list(df.columns)}"
        assert "cantidad" in df.columns
    
    print("✅ test_proporcion_aprovechable")


def test_volumen_por_dia():
    """volumen_por_dia devuelve DataFrame diario."""
    df = volumen_por_dia()
    assert df is not None, "Debe devolver un DataFrame"
    
    # Debería tener estas columnas
    if not df.empty:
        assert "dia" in df.columns
        assert "tipo" in df.columns
    
    print("✅ test_volumen_por_dia")


def test_datos_validacion():
    """datos_validacion devuelve pares (etiqueta_real, material)."""
    df = datos_validacion()
    # Puede devolver vacío si no hay etiquetas reales
    if not df.empty:
        assert "etiqueta_real" in df.columns
        assert "material" in df.columns
    
    print("✅ test_datos_validacion")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
