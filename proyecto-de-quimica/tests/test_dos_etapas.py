"""Motor de 2 etapas (pesos reales de models/) y utilidades offline de datos/descargar.py."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "datos"))

from motor_dos_etapas import MotorDosEtapas, contenedor_de_material_ui  # noqa: E402
from vision_engine import MOTOR_DOS_ETAPAS, MOTORES_PRODUCCION, ClasificadorResiduos  # noqa: E402

PESOS = (RAIZ / "models" / "detector_residuo.onnx").exists() and \
    (RAIZ / "models" / "clasificador_material.onnx").exists()
requiere_pesos = pytest.mark.skipif(not PESOS, reason="Faltan los ONNX del pipeline de 2 etapas")


@pytest.fixture(scope="module")
def motor():
    return MotorDosEtapas()


def test_dos_etapas_es_el_motor_por_defecto():
    assert MOTORES_PRODUCCION[0] == MOTOR_DOS_ETAPAS


@requiere_pesos
def test_carga_redes_reales(motor):
    assert motor.disponible, motor.error_carga
    assert len(motor.clases) == 16


@requiere_pesos
def test_pared_blanca_no_es_un_residuo(motor):
    assert motor.detectar_y_clasificar(np.full((480, 640, 3), 235, np.uint8)) == []
    r = motor.predecir(np.full((480, 640, 3), 235, np.uint8))
    assert r.material == "No identificado" and r.contenedor == "Negro"


@requiere_pesos
def test_salida_bien_formada(motor):
    img = np.full((480, 640, 3), 60, np.uint8)
    cv2.rectangle(img, (250, 120), (390, 420), (40, 40, 200), -1)
    for d in motor.detectar_y_clasificar(img, sin_detector_usar_imagen=True):
        assert d.contenedor in {"Blanco", "Negro", "Verde"}
        assert 0.0 <= d.probabilidad <= 1.0
        x1, y1, x2, y2 = d.caja_px
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480


@requiere_pesos
def test_clasificador_residuos_enruta_al_motor_nuevo():
    r = ClasificadorResiduos().predecir(np.full((480, 640, 3), 235, np.uint8), motor=MOTOR_DOS_ETAPAS)
    assert r.metodo == "Motor 2 Etapas"


def test_contenedor_por_nombre_ui():
    assert contenedor_de_material_ui("Icopor") == "Negro"
    assert contenedor_de_material_ui("Orgánico") == "Verde"
    assert contenedor_de_material_ui("Metal (lata)") == "Blanco"
    assert contenedor_de_material_ui("Algo inventado") is None


def test_descargar_une_etiquetas_de_empaque_sin_duplicar():
    import descargar
    prod = {"packaging_tags": ["en:bottle", "en:plastic"],
            "packaging_materials_tags": ["en:pet-1-polyethylene-terephthalate", "en:plastic"],
            "packaging_shapes_tags": ["en:bottle"]}
    assert descargar._unir_tags(prod) == ["en:bottle", "en:plastic",
                                          "en:pet-1-polyethylene-terephthalate"]


def test_descargar_etiquetas_combinadas_mapean_a_pet():
    import descargar
    from taxonomia import Taxonomia
    prod = {"packaging_materials_tags": ["en:pet-1-polyethylene-terephthalate"],
            "packaging_shapes_tags": ["en:bottle"]}
    res = Taxonomia.cargar().resolver_off(descargar._unir_tags(prod), [])
    assert (res.estado, res.clase) == ("mapeada", "pet_botella")


def test_descargar_exige_una_fuente(capsys):
    import descargar
    with pytest.raises(SystemExit):
        descargar.main([])
