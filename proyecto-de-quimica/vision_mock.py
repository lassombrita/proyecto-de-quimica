"""
vision_mock.py — Enrutador / Compatibilidad hacia vision_engine.py
==================================================================
Mantiene compatibilidad hacia atrás con scripts anteriores que importen
desde `vision_mock`. Delega toda la lógica en `vision_engine.py`.
"""

from vision_engine import (
    CATEGORIAS,
    COLORES_CONTENEDOR,
    ClasificadorResiduos,
    Morfologia,
    ResultadoClasificacion,
)

__all__ = [
    "CATEGORIAS",
    "COLORES_CONTENEDOR",
    "ClasificadorResiduos",
    "Morfologia",
    "ResultadoClasificacion",
]
