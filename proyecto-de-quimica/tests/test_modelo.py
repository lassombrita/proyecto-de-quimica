"""Pruebas del modelo ONNX y del motor de visión (se omiten si falta el modelo)."""

import cv2
import numpy as np
import pytest

from vision_engine import ClasificadorResiduos, CATEGORIAS

MOTORES = ["Motor Híbrido Estricto", "Motor IA (MobileNetV2)",
           "Motor Geométrico (Forma/Contornos)", "Motor Básico (Color)"]


def _imagen():
    img = np.full((360, 480, 3), 200, np.uint8)
    cv2.rectangle(img, (170, 60), (310, 300), (60, 60, 180), -1)
    return img


def test_softmax_de_la_red(requiere_modelo):
    """Las probabilidades (softmax de los logits) son >= 0 y suman 1."""
    red = cv2.dnn.readNetFromONNX(str(requiere_modelo))
    rgb = cv2.resize(cv2.cvtColor(_imagen(), cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255
    rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    red.setInput(rgb.transpose(2, 0, 1)[None].astype(np.float32))
    logits = red.forward().flatten()
    assert logits.shape == (6,)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    assert (p >= 0).all() and abs(p.sum() - 1) < 1e-5


def test_clasificador_carga(requiere_modelo):
    c = ClasificadorResiduos()
    assert c.red_onnx is not None and c.error_carga_dnn is None


@pytest.mark.parametrize("motor", MOTORES)
def test_los_cuatro_motores(requiere_modelo, motor):
    r = ClasificadorResiduos(modo_didactico=True).predecir(_imagen(), motor=motor)
    assert r.material in CATEGORIAS
    assert 0 <= r.confianza <= 100
    assert r.contenedor in ("Blanco", "Negro", "Verde")
    assert r.metodo == motor or motor == "Motor Híbrido Estricto"
