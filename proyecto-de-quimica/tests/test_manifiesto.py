"""Pruebas del manifiesto común (datos/construir_manifiesto.py) con datos sintéticos."""

import json

import cv2
import numpy as np
import pandas as pd

from datos import construir_manifiesto as cm


def _jpg(ruta, valor=128):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta), np.full((60, 80, 3), valor, np.uint8))


def test_manifiesto_multifuente(coco_sintetico, tmp_path):
    # TrashNet sintético: 10 fotos por carpeta
    tn = tmp_path / "trashnet"
    for c in cm.CARPETAS_TRASHNET:
        for i in range(10):
            _jpg(tn / c / f"{c}{i}.jpg")
    # Fotos propias con sesiones
    pr = tmp_path / "propias"
    _jpg(pr / "sesion_cocina" / "bolsa_blanca" / "a.jpg")
    _jpg(pr / "sesion_cocina" / "servilleta" / "b.jpg")
    _jpg(pr / "prueba_salon" / "icopor" / "c.jpg")
    # Open Food Facts sintético
    off = tmp_path / "off"
    (off / "img").mkdir(parents=True)
    _jpg(off / "img" / "7702001_1.jpg")
    _jpg(off / "img" / "7702002_1.jpg")
    productos = [
        {"code": "7702001", "countries_tags": ["en:colombia"],
         "packaging_tags": ["en:bag"], "categories_tags": ["en:snacks"]},
        {"code": "7702002", "countries_tags": ["en:france"],
         "packaging_tags": ["en:can"], "categories_tags": []},
    ]
    (off / "productos.jsonl").write_text("\n".join(json.dumps(p) for p in productos), encoding="utf-8")

    salida = tmp_path / "manifiesto.csv"
    assert cm.main(["--taco", str(coco_sintetico), "--taco-val", "", "--taco-prueba", "2",
                    "--trashnet", str(tn), "--propias", str(pr),
                    "--off-productos", str(off / "productos.jsonl"), "--off-imagenes", str(off / "img"),
                    "--salida", str(salida)]) == 0
    df = pd.read_csv(salida)

    assert set(df["fuente"]) == {"taco", "trashnet", "propias", "off"}
    assert df["id_objeto"].is_unique
    # TACO: cajas para el detector (la de 10 px también: el detector sí aprende objetos
    # pequeños; el clasificador las filtra por tamaño al exportar recortes); lote 2 = prueba
    taco = df[df["fuente"] == "taco"]
    assert taco["uso_detector"].all() and len(taco) == 9
    assert set(taco.loc[taco["ruta"].str.contains("batch_2"), "split"]) == {"prueba"}
    # La categoría inventada sigue sirviendo al detector pero no al clasificador
    rara = taco[taco["etiqueta_origen"] == "Categoria inventada"]
    assert (rara["estado_mapeo"] == "sin_mapeo").all() and not rara["uso_clasificador"].any()
    # TrashNet: plastic/metal/trash ambiguas; split 70/15/15
    tn_df = df[df["fuente"] == "trashnet"]
    assert set(tn_df.loc[tn_df["etiqueta_origen"] == "plastic", "estado_mapeo"]) == {"ambigua"}
    assert set(tn_df["split"]) == {"entrenamiento", "validacion", "prueba"}
    # Propias: la sesión "prueba_*" va a prueba
    pro = df[df["fuente"] == "propias"].set_index("etiqueta_origen")
    assert pro.loc["icopor", "split"] == "prueba"
    assert pro.loc["bolsa_blanca", "clase_master"] == "plastico_flexible_bolsa"
    assert pro.loc["servilleta", "clase_master"] == "sanitario_y_rechazo"
    # OFF: solo Colombia, nunca en prueba
    o = df[df["fuente"] == "off"]
    assert list(o["clase_master"]) == ["multicapa_flexible"] and "prueba" not in set(o["split"])
    # Reporte escrito al lado del manifiesto
    assert (tmp_path / "manifiesto_reporte.txt").read_text(encoding="utf-8").startswith("Manifiesto")


def test_sin_fugas_entre_conjuntos():
    tax = cm.Taxonomia.cargar()
    def filas():
        for split in ("entrenamiento", "prueba"):
            yield cm._fila(tax, fuente="propias", id_objeto=f"x{split}", ruta=cm.Path("a.jpg"),
                           etiqueta="icopor", supercategoria=None, split=split, grupo="misma_sesion")
    try:
        cm.construir(tax, [filas()])
    except RuntimeError as exc:
        assert "partidos" in str(exc)
    else:
        raise AssertionError("Debió detectar la fuga")
