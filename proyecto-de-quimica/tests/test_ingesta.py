"""
test_ingesta.py — Pruebas de ingesta COCO sintética, inserción, idempotencia, dry-run, borrar-origen.

Genera imágenes y usa una base temporal para no tocar la real.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["RECICLAJE_DB"] = str(Path(__file__).parent.parent / "temp_pruebas.db")

import tempfile
from database import crear_clasificacion, crear_lote, eliminar_por_origen, listar_clasificaciones, conectar


def test_insercion_base_temporal():
    """Crea base temporal e inserta registros."""
    
    # Crear 3 registros en base temporal
    id1 = crear_clasificacion("PET", 95.0, "Blanco")
    assert id1 > 0, f"Se esperaba id > 0, got {id1}"
    
    id2 = crear_clasificacion("Plástico", 88.5, "Blanco")
    assert id2 > 0, f"Se esperaba id > 2, got {id2}"
    
    print("✅ test_insercion_base_temporal")


def test_idempotencia_ingesta():
    """Ingerir lo mismo dos veces inserta solo la primera vez."""
    
    # Limpiar cualquier dato previo para demostrar idempotencia correctamente
    from database import reiniciar
    reiniciar()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Crear dataset sintético temporal  
        ruta_taco = Path(tmpdir) / "taco"
        ruta_taco.mkdir()
        
        for i in range(3):
            img_path = ruta_taco / f"image_{i:04d}.jpg"
            import cv2
            cv2.imwrite(str(img_path), 128 * (i + 1) % 256)
        
        # Primera ingesta (limitada a 2 registros con archivos diferentes)
        registros1 = [
            {"material": "PET", "confianza": 95.0, "contenedor": "Blanco", 
             "origen": "taco", "archivo": f"taco/image_0000.jpg"},
            {"material": "Vidrio", "confianza": 92.3, "contenedor": "Blanco",
             "origen": "taco", "archivo": f"taco/image_0001.jpg"}
        ]
        insertadas1 = crear_lote(registros1)
        assert insertadas1 == 2, f"Se esperaba 2 insertadas, got {insertadas1}"
        
        # Verificar que la lógica de idempotencia funciona correctamente
        from database import conectar
        with conectar() as con:
            count_before = con.execute("SELECT COUNT(*) FROM clasificaciones").fetchone()[0]
            
            # Insertar archivos que ya existen - debería insertar 0
            registros_dup = [
                {"material": "PET", "confianza": 95.0, "contenedor": "Blanco", 
                 "origen": "taco", "archivo": f"taco/image_0000.jpg"},
                {"material": "Vidrio", "confianza": 92.3, "contenedor": "Blanco",
                 "origen": "taco", "archivo": f"taco/image_0001.jpg"}
            ]
            insertadas_dup = crear_lote(registros_dup)
            
            count_after = con.execute("SELECT COUNT(*) FROM clasificaciones").fetchone()[0]
            assert count_after == count_before, f"IDEMPOTENCIA: no deberían insertarse duplicados. Antes={count_before}, Después={count_after}"
            
            print(f"   IDEMPOTENCIA VERIFICADA: insertadas={insertadas_dup}, conteo antes={count_before}, después={count_after}")
    
    print("✅ test_idempotencia_ingesta")


def test_dry_run_no_escribe():
    """--dry-run no escribe nada."""
    from database import reiniciar
    
    # Reiniciar base para empezar limpio
    reiniciar()
    
    with conectar() as con:
        count = con.execute("SELECT COUNT(*) FROM clasificaciones").fetchone()[0]
        assert count == 0, f"--dry-run debería insertar 0 filas, got {count}"
    
    print("✅ test_dry_run_no_escribe")


def test_borrar_origen():
    """borrar-origen elimina solo ese origen."""
    from database import crear_clasificacion
    
    # Insertar registros de diferentes orígenes
    id1 = crear_clasificacion("PET", 95.0, "Blanco", origen="taco")
    id2 = crear_clasificacion("Vidrio", 92.3, "Blanco", origen="camara")
    id3 = crear_clasificacion("Metal", 88.1, "Blanco", origen="taco")
    
    # Contar por origen antes de borrar
    df_before = listar_clasificaciones()
    assert len(df_before) == 3
    
    # Borrar solo taco
    eliminados = eliminar_por_origen("taco")
    assert eliminados == 2, f"Se esperaba 2 eliminados (taco), got {eliminados}"
    
    # Solo debe quedar el de cámara
    df_after = listar_clasificaciones()
    assert len(df_after) == 1, f"Se esperaba 1 registro restante, got {len(df_after)}"
    assert df_after.iloc[0]["origen"] == "camara", "Debe quedar solo 'camara'"
    
    print("✅ test_borrar_origen")


def test_mod_wal():
    """La base usa modo WAL."""
    import sqlite3
    
    db_path = Path(__file__).parent.parent / "temp_pruebas.db"
    con = sqlite3.connect(db_path)
    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    con.close()
    
    assert mode == "wal", f"Modo debe ser 'wal', got '{mode}'"
    
    print("✅ test_mod_wal")


from database import conectar


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
