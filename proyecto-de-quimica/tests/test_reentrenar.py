"""Pruebas de entrenamiento/reentrenar.py que no requieren datasets descargados."""

import sys

import numpy as np

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "entrenamiento"))
import reentrenar as rt  # noqa: E402


def _imagenes(n=3):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (200, 260, 3), dtype=np.uint8) for _ in range(n)]


def test_normalizar_lotes():
    assert rt.normalizar_lotes("14, 15,test") == {"batch_14", "batch_15", "test"}


def test_backbone_mas_cabeza_igual_al_modelo(requiere_modelo, tmp_path):
    """Separar red base + capa final reproduce exactamente el modelo original."""
    ruta, cabeza = rt.extraer_backbone(requiere_modelo, tmp_path / "backbone.onnx")
    extractor = rt.Extractor(ruta, tam_lote=4)
    assert extractor.caracteristicas(_imagenes(2)).shape == (2, cabeza.W.shape[1])
    assert rt.verificar_equivalencia(requiere_modelo, cabeza, extractor, _imagenes()) < 1e-3


def test_exportar_modelo_con_cabeza_nueva(requiere_modelo, tmp_path):
    ruta, cabeza = rt.extraer_backbone(requiere_modelo, tmp_path / "backbone.onnx")
    extractor = rt.Extractor(ruta)
    rng = np.random.default_rng(1)
    nueva = rt.Cabeza(rng.normal(size=cabeza.W.shape).astype(np.float32),
                      rng.normal(size=cabeza.b.shape).astype(np.float32), "", "", "")
    salida = rt.exportar_modelo(requiere_modelo, nueva, tmp_path / "nuevo.onnx")
    assert rt.verificar_equivalencia(salida, nueva, extractor, _imagenes()) < 1e-3


def test_decision_rechaza_si_empeora_trashnet():
    m = lambda r, e: {"recall_grupos": r, "exactitud": e}
    res = {"taco": {c: {"actual": m(0.3, 0.3), "nuevo": m(0.6, 0.6)} for c in ("limpia", "webcam")},
           "trashnet": {"limpia": {"actual": m(0.9, 0.9), "nuevo": m(0.8, 0.8)}}}
    aceptado, _ = rt.decidir(res, tolerancia=0.03)
    assert aceptado is False
    res["trashnet"]["limpia"]["nuevo"] = m(0.89, 0.89)
    assert rt.decidir(res, tolerancia=0.03)[0] is True
