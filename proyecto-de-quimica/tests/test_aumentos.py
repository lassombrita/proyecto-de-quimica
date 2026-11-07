"""Pruebas del aumento de datos (albumentations) y de la simulación de webcam."""

import albumentations as A
import numpy as np
import pytest

import ingesta_publica as ip


@pytest.fixture
def muestra_multicapa():
    return ip.Muestra(archivo="batch_1/000.jpg#1", ruta=None, bbox=None, tam_anotado=None,
                      clase=ip.CLASES_NORMATIVAS["No aprovechable (Multicapa)"],
                      detalle="Crisp packet", critica=True)


@pytest.fixture
def imagen():
    rng = np.random.default_rng(1)
    img = np.full((400, 300, 3), 120, np.uint8)
    img[60:340, 50:250] = (40, 40, 200)
    return np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)


def _variantes(muestra, imagen, semilla):
    return [v for _, v in ip.generar_variantes(muestra, imagen, aumentos=3, incluir_original=True,
                                               semilla=semilla, calidad_jpeg=75)]


def test_reproducible_con_la_misma_semilla(muestra_multicapa, imagen):
    a = _variantes(muestra_multicapa, imagen, semilla=1)
    b = _variantes(muestra_multicapa, imagen, semilla=1)
    assert len(a) == 4
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_semillas_distintas_dan_variantes_distintas(muestra_multicapa, imagen):
    a = _variantes(muestra_multicapa, imagen, semilla=1)
    b = _variantes(muestra_multicapa, imagen, semilla=2)
    assert any(not np.array_equal(x, y) for x, y in zip(a[1:], b[1:]))


def test_tubería_contiene_las_transformaciones_pedidas():
    aumentador = ip.construir_aumentador(400, 300, metalizada=True, semilla=0)
    tipos = {type(t) for t in aumentador.transforms}
    for requerida in (A.RandomCrop, A.RandomSunFlare, A.RandomBrightnessContrast,
                      A.ImageCompression):
        assert requerida in tipos
    destello = next(t for t in aumentador.transforms if isinstance(t, A.RandomSunFlare))
    assert destello.p == pytest.approx(0.8)          # Más destellos en metalizados
    normal = ip.construir_aumentador(400, 300, metalizada=False, semilla=0)
    assert next(t for t in normal.transforms if isinstance(t, A.RandomSunFlare)).p == pytest.approx(0.3)


def test_recorte_conserva_el_88_por_ciento(muestra_multicapa, imagen):
    for _, v in list(ip.generar_variantes(muestra_multicapa, imagen, aumentos=3,
                                          incluir_original=False, semilla=3, calidad_jpeg=75)):
        assert v.shape[0] == int(400 * 0.88) and v.shape[1] == int(300 * 0.88)
        assert v.dtype == np.uint8 and v.shape[2] == 3


def test_simular_camara_limita_a_640(imagen):
    grande = np.repeat(np.repeat(imagen, 4, axis=0), 4, axis=1)       # 1600 x 1200
    assert max(ip.simular_camara(grande).shape[:2]) == 640
    assert ip.simular_camara(imagen).shape == imagen.shape            # No agranda


def test_recortar_con_margen_y_caja_invalida(imagen):
    r = ip.recortar(imagen, (50, 60, 200, 280), margen=0.30)
    assert r.shape[:2] == (min(400, 60 + 280 + 84) - max(0, 60 - 84), 300)
    with pytest.raises(ValueError):
        ip.recortar(imagen, (1000, 1000, 50, 50), margen=0.1)
