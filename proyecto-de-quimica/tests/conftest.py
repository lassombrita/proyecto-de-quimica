"""
conftest.py — Configuración compartida de las pruebas (pytest)
==============================================================
* Cada prueba usa su PROPIA base SQLite temporal: nunca se toca data/reciclaje.db
  y ninguna prueba ve datos de otra (aislamiento total).
* `coco_sintetico` genera un mini-dataset en formato MS COCO (estilo TACO) con las
  clases críticas, para probar la ingesta sin descargar nada.

Ejecutar desde la carpeta del proyecto:
    python -m pytest -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import database as db  # noqa: E402

MODELO = RAIZ / "models" / "trashnet_mobilenetv2.onnx"

CATEGORIAS_SINTETICAS = [
    {"id": 1, "name": "Crisp packet", "supercategory": "Plastic bag & wrapper"},
    {"id": 2, "name": "Foam food container", "supercategory": "Plastic container"},
    {"id": 3, "name": "Aerosol", "supercategory": "Can"},
    {"id": 4, "name": "Clear plastic bottle", "supercategory": "Bottle"},
    {"id": 5, "name": "Food waste", "supercategory": "Food waste"},
    {"id": 6, "name": "Categoria inventada", "supercategory": "Rara"},
]


@pytest.fixture(autouse=True)
def bd_temporal(tmp_path, monkeypatch):
    """Redirige `database` a una base nueva y vacía para CADA prueba."""
    monkeypatch.setattr(db, "RUTA_BD", tmp_path / "prueba.db")
    db._inicializada.clear()
    yield tmp_path / "prueba.db"
    db._inicializada.clear()


def _imagen_objeto(color: tuple[int, int, int], rng: np.random.Generator,
                   alto: int = 480, ancho: int = 640) -> np.ndarray:
    """Fondo gris con un rectángulo de color (el «residuo») en la caja (160,120,240,200)."""
    img = np.full((alto, ancho, 3), 110, np.uint8)
    cv2.rectangle(img, (160, 120), (400, 320), color, -1)
    ruido = rng.normal(0, 4, img.shape)
    return np.clip(img + ruido, 0, 255).astype(np.uint8)


@pytest.fixture
def coco_sintetico(tmp_path) -> Path:
    """
    Crea <tmp>/taco/annotations.json + batch_1/ y batch_2/ con 8 fotos.

    Incluye: las 3 clases críticas, una clase no evaluable (Food waste), una categoría
    sin mapeo, una caja diminuta (debe descartarse) y una foto con orientación EXIF.
    """
    rng = np.random.default_rng(0)
    raiz = tmp_path / "taco"
    colores = [(30, 30, 200), (240, 240, 240), (170, 170, 170), (200, 120, 40)]
    imagenes, anotaciones, aid = [], [], 1
    for i in range(8):
        lote = f"batch_{1 + i % 2}"
        (raiz / lote).mkdir(parents=True, exist_ok=True)
        nombre = f"{lote}/{i:03d}.jpg"
        bgr = _imagen_objeto(colores[i % 4], rng)
        pil = Image.fromarray(bgr[:, :, ::-1])
        if i == 0:
            # Píxeles guardados «acostados» + EXIF orientación 6: el JSON describe la
            # imagen ya rotada (640x480), como en muchas fotos de celular de TACO.
            exif = Image.Exif()
            exif[0x0112] = 6
            pil.transpose(Image.Transpose.ROTATE_90).save(raiz / nombre, exif=exif)
        else:
            pil.save(raiz / nombre)
        imagenes.append({"id": i + 1, "file_name": nombre, "width": 640, "height": 480})
        categoria = CATEGORIAS_SINTETICAS[i % len(CATEGORIAS_SINTETICAS)]["id"]
        anotaciones.append({"id": aid, "image_id": i + 1, "category_id": categoria,
                            "bbox": [160, 120, 240, 200], "area": 48000, "iscrowd": 0,
                            "segmentation": []})
        aid += 1
    # Caja diminuta (debe ignorarse por --lado-minimo)
    anotaciones.append({"id": aid, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10],
                        "area": 100, "iscrowd": 0, "segmentation": []})
    (raiz / "annotations.json").write_text(json.dumps({
        "images": imagenes, "annotations": anotaciones, "categories": CATEGORIAS_SINTETICAS}),
        encoding="utf-8")
    return raiz


@pytest.fixture
def requiere_modelo():
    """Omite la prueba si el modelo ONNX no está en models/."""
    if not MODELO.is_file():
        pytest.skip("models/trashnet_mobilenetv2.onnx no está disponible")
    return MODELO
