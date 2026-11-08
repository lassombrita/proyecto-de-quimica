"""Pruebas de las partes del pipeline que no requieren PyTorch ni Ultralytics."""

import numpy as np
import pandas as pd
import pytest
import yaml

from entrenamiento import train_pipeline as tp


def test_config_por_defecto_es_valida():
    cfg = yaml.safe_load((tp.RAIZ / "entrenamiento" / "pipeline.yaml").read_text(encoding="utf-8"))
    assert cfg["detector"]["modelo"] in ("yolo11n.pt", "yolo26n.pt")
    assert cfg["clasificador"]["arquitectura"] == "mobilenet_v3_large"
    assert cfg["clasificador"]["congelar_backbone"] is False
    criticas = set(cfg["evaluacion"]["clases_criticas"])
    assert criticas <= set(tp.Taxonomia.cargar().nombres)


def test_pesos_muestreo():
    y = np.array([0, 0, 0, 0, 1])
    assert np.allclose(tp.pesos_muestreo(y, "inversa"), [0.25] * 4 + [1.0])
    assert np.allclose(tp.pesos_muestreo(y, "uniforme"), 1.0)


def test_balanceo_topes_por_fuente_y_clase():
    filas = ([{"recorte": f"a{i}", "clase_id": 3, "split": "entrenamiento", "fuente": "off"} for i in range(100)]
             + [{"recorte": f"b{i}", "clase_id": 3, "split": "entrenamiento", "fuente": "taco"} for i in range(70)]
             + [{"recorte": f"c{i}", "clase_id": 7, "split": "entrenamiento", "fuente": "taco"} for i in range(50)]
             + [{"recorte": f"v{i}", "clase_id": 7, "split": "validacion", "fuente": "taco"} for i in range(9)])
    ent = tp.seleccionar_entrenamiento(pd.DataFrame(filas),
                                       {"tope_por_clase": 40, "tope_fraccion_fuente": {"off": 0.30}}, 0)
    assert set(ent["split"]) == {"entrenamiento"}
    assert (ent.groupby("clase_id").size() <= 40).all()
    c3 = ent[ent["clase_id"] == 3]
    # Antes del tope por clase: off <= 0.30 / 0.70 * 70 = 30 de 100
    assert (c3["fuente"] == "off").sum() <= 0.45 * len(c3)


def test_recortes_de_fondo_no_tocan_cajas():
    import random
    df = pd.DataFrame([{"ruta": "x.jpg", "fuente": "taco", "uso_detector": True, "ancho": 640, "alto": 480,
                        "bbox_x": 100, "bbox_y": 100, "bbox_w": 200, "bbox_h": 200,
                        "split": "entrenamiento", "grupo_split": "batch_1"}])
    fondos = tp._recortes_fondo(df, {"fuentes": ["taco"], "recortes_por_imagen": 5, "max_total": 50},
                                random.Random(0))
    assert fondos
    for _, (x, y, w, h), _, _ in fondos:
        assert x + w <= 100 or x >= 300 or y + h <= 100 or y >= 300


def test_dry_run_sin_manifiesto_falla_claro(tmp_path):
    cfg = yaml.safe_load((tp.RAIZ / "entrenamiento" / "pipeline.yaml").read_text(encoding="utf-8"))
    cfg["rutas"] = {k: str(tmp_path / k) for k in ("manifiesto", "trabajo", "pesos_base", "modelos")}
    ruta = tmp_path / "p.yaml"
    ruta.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(SystemExit, match="construir_manifiesto"):
        tp.main(["--config", str(ruta), "--dry-run", "--etapas", "exportar-detector"])


def test_etapas_desconocidas():
    with pytest.raises(SystemExit):
        tp.main(["--etapas", "entrenar-todo-magicamente"])
