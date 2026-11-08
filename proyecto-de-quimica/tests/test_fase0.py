"""
Pruebas de la Fase 0: sesgo «blanco = PET», plástico genérico, aislamiento de los
motores didácticos y conexión del modelo real en el escáner Edge.
"""

import time

import cv2
import numpy as np
import pytest

import database as db
from vision_engine import (
    CATEGORIAS,
    MAPEO_TRASHNET,
    MOTORES_DIDACTICOS,
    MOTORES_PRODUCCION,
    ClasificadorResiduos,
    Morfologia,
)


def _fondo(valor: int) -> np.ndarray:
    return np.full((480, 640, 3), valor, np.uint8)


def _mancha_blanca() -> np.ndarray:
    """Óvalo blanco con ruido: silueta típica de una bolsa arrugada (sin cuello)."""
    img = _fondo(120)
    cv2.ellipse(img, (320, 240), (150, 110), 0, 0, 360, (240, 240, 240), -1)
    ruido = np.random.default_rng(0).normal(0, 6, img.shape)
    return np.clip(img + ruido, 0, 255).astype(np.uint8)


def test_plastic_no_se_fuerza_a_pet():
    assert MAPEO_TRASHNET["plastic"] == "Plástico"
    assert CATEGORIAS["Plástico"]["contenedor"] == "Blanco"


@pytest.mark.parametrize("motor", MOTORES_DIDACTICOS)
def test_produccion_rechaza_motores_didacticos(motor):
    with pytest.raises(ValueError, match="didáctico"):
        ClasificadorResiduos().predecir(_fondo(200), motor=motor)


def test_motor_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        ClasificadorResiduos(modo_didactico=True).predecir(_fondo(200), motor="Motor Mágico")


def test_color_ya_no_convierte_blanco_en_pet():
    r = ClasificadorResiduos(modo_didactico=True).predecir(_fondo(235), motor="Motor Básico (Color)")
    assert r.material == "No identificado"


@pytest.mark.parametrize("motor", MOTORES_PRODUCCION)
def test_fondo_blanco_y_bolsa_no_son_pet(requiere_modelo, motor):
    c = ClasificadorResiduos()
    for img in (_fondo(235), _mancha_blanca()):
        assert c.predecir(img, motor=motor).material != "PET"


def test_ovalo_no_es_cuello_de_botella():
    """Un óvalo tiene ambos extremos angostos: no es evidencia de botella."""
    ovalo = Morfologia(relacion_aspecto=1.4, ratio_cuello=0.4, ratio_base=0.4)
    botella = Morfologia(relacion_aspecto=3.0, ratio_cuello=0.4, ratio_base=1.0)
    assert not ovalo.tiene_cuello
    assert botella.tiene_cuello


def test_silueta_de_botella_sigue_siendo_pet_en_geometria():
    img = _fondo(90)
    cv2.rectangle(img, (260, 80), (380, 420), (230, 230, 230), -1)
    cv2.rectangle(img, (295, 40), (345, 90), (230, 230, 230), -1)
    r = ClasificadorResiduos(modo_didactico=True).predecir(img, motor="Motor Geométrico (Forma/Contornos)")
    assert r.material == "PET"


def test_hibrido_no_suma_confianza(requiere_modelo):
    r = ClasificadorResiduos().predecir(_mancha_blanca(), motor="Motor Híbrido Estricto")
    assert r.top_predicciones and r.confianza == pytest.approx(r.top_predicciones[0][1], abs=0.11)


# --- Escáner Edge: modelo real + registrador asíncrono -----------------------
def test_modelo_real_edge(requiere_modelo):
    import vision_edge as ve
    m = ve.ModeloVisionReal(motor="Motor Híbrido Estricto")   # Ruta por contornos
    img = _fondo(60)
    cv2.rectangle(img, (200, 100), (420, 380), (230, 230, 230), -1)
    dets = m.inferir(img)
    assert len(dets) == 1
    d = dets[0]
    assert d.material in CATEGORIAS
    assert db.color_de_contenedor(d.contenedor) in ("BLANCO", "NEGRO", "VERDE")
    assert all(0.0 <= v <= 1.0 for v in d.caja)


def test_modelo_real_edge_dos_etapas(requiere_modelo):
    import vision_edge as ve
    m = ve.ModeloVisionReal()          # Por defecto: detector YOLO + MobileNetV3
    for d in m.inferir(_fondo(235)) + m.inferir(_mancha_blanca()):
        assert db.color_de_contenedor(d.contenedor) in ("BLANCO", "NEGRO", "VERDE")
        assert all(0.0 <= v <= 1.0 for v in d.caja)
    assert m.inferir(_fondo(235)) == []   # Pared blanca: no hay residuo


def test_edge_rechaza_motor_didactico():
    import vision_edge as ve
    with pytest.raises(ValueError):
        ve.ModeloVisionReal(motor="Motor Básico (Color)")


def test_registrador_asincrono_y_estadisticas():
    reg = db.RegistradorAsincrono(motor="prueba")
    assert reg.encolar_deteccion("Plástico", "BLANCO (Aprovechable)", 81.5)
    assert reg.encolar_deteccion("No identificado", "NEGRO (No Aprovechable)", 30.0)
    assert not reg.encolar_deteccion("X", "MORADO", 50.0)
    reg.detener()
    assert reg.metricas()["escritos"] == 2
    stats = db.get_stats("minuto")
    assert stats["total"] == 2
    assert set(stats["por_contenedor"]["color"]) == {"BLANCO", "NEGRO"}
    assert len(db.get_events(10)) == 2
    assert db.reset_eventos() == 2


def test_color_de_contenedor():
    assert db.color_de_contenedor("VERDE (Orgánico)") == "VERDE"
    assert db.contenedor_canonico("NEGRO (No Aprovechable - Material de Rechazo)") == "Negro"
    with pytest.raises(ValueError):
        db.color_de_contenedor("Azul")
