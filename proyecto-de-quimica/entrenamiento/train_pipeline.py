"""
train_pipeline.py — Entrenamiento secuencial: detector "residuo" + clasificador de material
==========================================================================================
Un solo comando, configurado por `entrenamiento/pipeline.yaml`, que consume el
manifiesto común (`datos/construir_manifiesto.py`) y produce dos modelos ONNX listos
para inferencia OFFLINE:

    models/detector_residuo.onnx        YOLO11n · 1 clase ("residuo")
    models/clasificador_material.onnx   MobileNetV3-Large · 16 clases (taxonomia/master.yaml)
    models/clasificador_material.json   clases, temperatura, normalización, versión de taxonomía

Etapas (en orden; se pueden elegir con --etapas):

    preparar-pesos         Descarga UNA vez yolo11n.pt y MobileNetV3 ImageNet (única etapa online)
    exportar-detector      Manifiesto -> dataset YOLO (images/ + labels/ + data.yaml), 1 clase
    entrenar-detector      Ultralytics YOLO .train() + export ONNX
    exportar-recortes      Manifiesto -> recortes 256 px por clase + recortes de fondo (clase 15)
    entrenar-clasificador  MobileNetV3-Large, fine-tuning completo, sampler balanceado
    calibrar               Temperatura (Guo et al., 2017) sobre validación
    exportar-clasificador  ONNX + verificación con onnxruntime y cv2.dnn
    evaluar                Prueba: recall por clase, caneca correcta, contaminación de la Blanca

Ejemplos:
    python entrenamiento/train_pipeline.py --etapas preparar-pesos          # con internet, 1 vez
    python entrenamiento/train_pipeline.py                                   # todo, offline
    python entrenamiento/train_pipeline.py --etapas exportar-recortes,entrenar-clasificador
    python entrenamiento/train_pipeline.py --dry-run                         # plan y conteos
    python entrenamiento/train_pipeline.py --rapido                          # humo: 1 época, pocas imágenes

Requisitos:  pip install -r entrenamiento/requirements-entrenamiento.txt
             (torch, torchvision, ultralytics, onnx, onnxruntime, pyarrow)

Licencias: Ultralytics YOLO es AGPL-3.0 (el repositorio debe seguir siendo público).
ZeroWaste es CC BY-NC 4.0: los pesos que lo usen no deben comercializarse.

Autor: Johan Lasso
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from datos.construir_manifiesto import cargar_manifiesto  # noqa: E402
from taxonomia import Taxonomia  # noqa: E402

ETAPAS = ("preparar-pesos", "exportar-detector", "entrenar-detector", "exportar-recortes",
          "entrenar-clasificador", "calibrar", "exportar-clasificador", "evaluar")
ETAPAS_POR_DEFECTO = ETAPAS[1:]          # preparar-pesos se pide explícitamente (usa internet)
SPLIT_YOLO = {"entrenamiento": "train", "validacion": "val", "prueba": "test"}
MEDIA = (0.485, 0.456, 0.406)
DESV = (0.229, 0.224, 0.225)


# ===========================================================================
# 0. CONFIGURACIÓN Y CONTEXTO
# ===========================================================================
@dataclass
class Contexto:
    cfg: dict[str, Any]
    tax: Taxonomia
    trabajo: Path
    pesos_base: Path
    modelos: Path
    manifiesto: Path
    rapido: bool = False
    dry_run: bool = False

    @property
    def dir_detector(self) -> Path:
        return self.trabajo / "detector"

    @property
    def dir_recortes(self) -> Path:
        return self.trabajo / "recortes"

    @property
    def dir_clasificador(self) -> Path:
        return self.trabajo / "clasificador"


def _ruta(valor: str) -> Path:
    p = Path(valor)
    return p if p.is_absolute() else RAIZ / p


def cargar_contexto(ruta_cfg: Path, rapido: bool, dry_run: bool) -> Contexto:
    cfg = yaml.safe_load(Path(ruta_cfg).read_text(encoding="utf-8"))
    r = cfg["rutas"]
    ctx = Contexto(cfg, Taxonomia.cargar(), _ruta(r["trabajo"]), _ruta(r["pesos_base"]),
                   _ruta(r["modelos"]), _ruta(r["manifiesto"]), rapido, dry_run)
    if rapido:  # Prueba de humo: todo el flujo en minutos
        cfg["detector"]["entrenamiento"].update(epochs=1, imgsz=320, batch=4, workers=0, cache=False)
        cfg["detector"]["exportar"]["imgsz"] = 320
        cfg["clasificador"]["entrenamiento"].update(epochs=1, batch=8, workers=0, warmup_epochs=0)
        cfg["clasificador"]["fondo"]["max_total"] = 20
    # Las librerías guardan/leen sus pesos aquí: después de preparar-pesos no se usa la red.
    os.environ.setdefault("TORCH_HOME", str(ctx.pesos_base / "torch"))
    if cfg.get("offline", True):
        # Ultralytics deja de intentar descargas (fuentes, chequeos de versión).
        os.environ.setdefault("YOLO_OFFLINE", "true")
    random.seed(cfg.get("semilla", 42))
    np.random.seed(cfg.get("semilla", 42))
    return ctx


def log(msg: str) -> None:
    # Convert box-drawing chars and emojis to ASCII for Windows CP1252 console
    safe_msg = msg.replace('\u2500', '-').replace('\u2501', '|').replace('\u274c', 'X').replace('\u2705', 'OK').replace('\u2714', 'OK')
    print(f"[{time.strftime('%H:%M:%S')}] {safe_msg}", flush=True)


def _nombre_seguro(id_objeto: str, ruta: str) -> str:
    """Nombre de archivo único y corto por objeto/imagen."""
    prefijo = id_objeto.split(":")[0]
    prefijo = "".join(c for c in prefijo if c.isalnum() or c in ("-", "_")) or "img"
    h = hashlib.sha1(id_objeto.encode()).hexdigest()[:16]
    sufijo = Path(ruta).suffix.lower() or ".jpg"
    return f"{prefijo}_{h}{sufijo}"


def leer_imagen_pil(ruta: str | Path):
    """Imagen RGB con la orientación EXIF aplicada (igual criterio que COCO/Ultralytics)."""
    from PIL import Image, ImageOps
    with Image.open(ruta) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _manifiesto(ctx: Contexto) -> pd.DataFrame:
    if not ctx.manifiesto.exists() and not ctx.manifiesto.with_suffix(".csv").exists():
        sys.exit(f"❌ No existe {ctx.manifiesto}. Ejecute primero datos/construir_manifiesto.py")
    df = cargar_manifiesto(ctx.manifiesto)
    if ctx.rapido:
        # Barajar + head() por grupo: compatible con pandas 2 y 3 (apply quita la columna)
        df = df.sample(frac=1.0, random_state=0).groupby(["fuente", "split"]).head(40)
    return df


# ===========================================================================
# 1. PREPARAR PESOS (única etapa con internet)
# ===========================================================================
def etapa_preparar_pesos(ctx: Contexto) -> None:
    ctx.pesos_base.mkdir(parents=True, exist_ok=True)
    os.environ.pop("YOLO_OFFLINE", None)
    destino = ctx.pesos_base / ctx.cfg["detector"]["modelo"]
    if destino.exists():
        log(f"✔ {destino.name} ya existe")
    else:
        from ultralytics import YOLO
        cwd = Path.cwd()
        os.chdir(ctx.pesos_base)          # Ultralytics descarga al directorio actual
        try:
            YOLO(ctx.cfg["detector"]["modelo"])
        finally:
            os.chdir(cwd)
        log(f"✔ Descargado {destino}")
    if ctx.cfg["clasificador"].get("preentrenado", True):
        from torchvision.models import MobileNet_V3_Large_Weights
        MobileNet_V3_Large_Weights.IMAGENET1K_V2.get_state_dict(progress=True)
        log(f"✔ MobileNetV3-Large ImageNet en {os.environ['TORCH_HOME']}")


# ===========================================================================
# 2. DETECTOR: exportar dataset YOLO y entrenar
# ===========================================================================
def _enlazar(origen: Path, destino: Path, enlazar: bool) -> None:
    if destino.exists():
        return
    if not origen.exists():
        raise FileNotFoundError(f"No existe la imagen de origen: {origen}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    if enlazar:
        try:
            destino.symlink_to(origen)
            return
        except OSError:  # Windows sin modo desarrollador
            pass
    shutil.copy2(origen, destino)


def etapa_exportar_detector(ctx: Contexto) -> Path:
    cfg = ctx.cfg["detector"]
    df = _manifiesto(ctx)
    df = df[df["uso_detector"] & df["fuente"].isin(cfg["fuentes"])]
    if not cfg.get("incluir_sin_mapeo", True):
        df = df[df["estado_mapeo"] != "sin_mapeo"]
    df = df.dropna(subset=["bbox_x", "ancho", "alto"])
    log(f"Detector: {len(df):,} cajas en {df['ruta'].nunique():,} imágenes")
    if ctx.dry_run or df.empty:
        print(df.groupby(["fuente", "split"])["ruta"].nunique().to_string() if not df.empty else "  (vacío)")
        return ctx.dir_detector / "data.yaml"

    base = ctx.dir_detector
    if base.exists():
        shutil.rmtree(base)
    for split in SPLIT_YOLO.values():
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    for ruta, g in df.groupby("ruta"):
        split = SPLIT_YOLO[g["split"].iloc[0]]
        fuente = g["fuente"].iloc[0]
        nombre = _nombre_seguro(f"{fuente}:{ruta}", ruta)
        _enlazar(Path(ruta), base / "images" / split / nombre, cfg.get("enlazar_imagenes", True))
        W, H = float(g["ancho"].iloc[0]), float(g["alto"].iloc[0])
        lineas = []
        for _, f in g.iterrows():
            x, y, w, h = f["bbox_x"], f["bbox_y"], f["bbox_w"], f["bbox_h"]
            cx, cy = min(max((x + w / 2) / W, 0), 1), min(max((y + h / 2) / H, 0), 1)
            lineas.append(f"0 {cx:.6f} {cy:.6f} {min(w / W, 1):.6f} {min(h / H, 1):.6f}")
        (base / "labels" / split / Path(nombre).with_suffix(".txt").name).write_text(
            "\n".join(lineas) + "\n", encoding="utf-8")
    data = {"path": base.resolve().as_posix(), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {0: "residuo"}}
    (base / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    log(f"✔ Dataset YOLO en {base}")
    return base / "data.yaml"


def _dispositivo(valor: str | int | None) -> str:
    if valor not in (None, "auto"):
        return str(valor)
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def etapa_entrenar_detector(ctx: Contexto) -> Path | None:
    cfg = ctx.cfg["detector"]
    data = ctx.dir_detector / "data.yaml"
    pesos = ctx.pesos_base / cfg["modelo"]
    if ctx.dry_run:
        log(f"Entrenaría {cfg['modelo']} con {data} · {cfg['entrenamiento']}")
        return None
    if not data.exists():
        sys.exit("❌ Falta el dataset del detector: ejecute la etapa exportar-detector.")
    from ultralytics import YOLO
    origen = str(pesos) if pesos.exists() else cfg["modelo"]
    if not pesos.exists():
        log(f"⚠️  {pesos} no existe: Ultralytics intentará descargarlo (requiere internet).")
    params = dict(cfg["entrenamiento"])
    params["device"] = _dispositivo(params.get("device"))
    modelo = YOLO(origen)
    modelo.train(data=str(data), project=str(ctx.trabajo / "corridas"), name="detector",
                 exist_ok=True, seed=ctx.cfg.get("semilla", 42), verbose=False, **params)
    # trainer.best es estable entre versiones (el valor de retorno de train() no lo es).
    mejor = Path(modelo.trainer.best)
    if not mejor.exists():
        mejor = Path(modelo.trainer.save_dir) / "weights" / "last.pt"
    exp = cfg["exportar"]
    onnx = YOLO(str(mejor)).export(format=exp["formato"], imgsz=exp["imgsz"],
                                   simplify=exp.get("simplify", True), dynamic=exp.get("dynamic", False),
                                   half=exp.get("half", False))
    destino = ctx.modelos / exp["salida"]
    ctx.modelos.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx, destino)
    meta = {"arquitectura": cfg["modelo"], "clases": ["residuo"], "imgsz": exp["imgsz"],
            "conf": cfg["inferencia"]["conf"], "iou": cfg["inferencia"]["iou"],
            "entrenado": time.strftime("%Y-%m-%d %H:%M"), "licencia": "AGPL-3.0 (Ultralytics)"}
    destino.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"✔ Detector exportado: {destino}")
    return destino


# ===========================================================================
# 3. CLASIFICADOR: recortes, dataset, entrenamiento
# ===========================================================================
def _recortar(im, fila, margen: float):
    if pd.isna(fila["bbox_x"]):
        return im
    W, H = im.size
    x, y, w, h = fila["bbox_x"], fila["bbox_y"], fila["bbox_w"], fila["bbox_h"]
    # Si el JSON declara otro tamaño (EXIF), escalar la caja al tamaño real
    if fila.get("ancho") and fila.get("alto") and (W, H) != (int(fila["ancho"]), int(fila["alto"])):
        sx, sy = W / float(fila["ancho"]), H / float(fila["alto"])
        x, y, w, h = x * sx, y * sy, w * sx, h * sy
    mx, my = w * margen, h * margen
    return im.crop((max(0, int(x - mx)), max(0, int(y - my)),
                    min(W, int(x + w + mx)), min(H, int(y + h + my))))


def _redimensionar(im, lado: int):
    escala = lado / min(im.size)
    if escala < 1:
        im = im.resize((max(1, round(im.size[0] * escala)), max(1, round(im.size[1] * escala))))
    return im


def _recortes_fondo(df: pd.DataFrame, cfg_fondo: dict, rng: random.Random) -> list[tuple[str, tuple, str, str]]:
    """Ventanas aleatorias que NO tocan ninguna caja: ejemplos de la clase fondo_sin_objeto."""
    salida = []
    imgs = df[df["fuente"].isin(cfg_fondo["fuentes"]) & df["uso_detector"]].groupby("ruta")
    for ruta, g in imgs:
        W, H = float(g["ancho"].iloc[0]), float(g["alto"].iloc[0])
        cajas = g[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(float)
        for _ in range(cfg_fondo["recortes_por_imagen"]):
            for _intento in range(15):
                lado = rng.uniform(0.15, 0.35) * min(W, H)
                x, y = rng.uniform(0, W - lado), rng.uniform(0, H - lado)
                libre = all(x + lado <= bx or bx + bw <= x or y + lado <= by or by + bh <= y
                            for bx, by, bw, bh in cajas)
                if libre:
                    salida.append((ruta, (x, y, lado, lado), g["split"].iloc[0], g["grupo_split"].iloc[0]))
                    break
        if len(salida) >= cfg_fondo["max_total"]:
            break
    return salida


def etapa_exportar_recortes(ctx: Contexto) -> Path:
    cfg = ctx.cfg["clasificador"]
    rc = cfg["recortes"]
    df = _manifiesto(ctx)
    usar = df["uso_clasificador"]
    if cfg.get("usar_ambiguas"):
        usar |= df["estado_mapeo"] == "ambigua"
    clf = df[usar].copy()
    pequenas = clf["bbox_w"].notna() & (clf[["bbox_w", "bbox_h"]].min(axis=1) < rc["lado_minimo_caja"])
    clf = clf[~pequenas]
    rng = random.Random(ctx.cfg.get("semilla", 42))
    fondos = _recortes_fondo(df, cfg["fondo"], rng)
    log(f"Clasificador: {len(clf):,} recortes etiquetados ({int(pequenas.sum()):,} cajas < "
        f"{rc['lado_minimo_caja']} px descartadas) + {len(fondos):,} de fondo")
    if ctx.dry_run:
        print(pd.crosstab(clf["clase_master"], clf["split"]).to_string() if not clf.empty else "  (vacío)")
        return ctx.dir_recortes / "indice.csv"

    base = ctx.dir_recortes
    base.mkdir(parents=True, exist_ok=True)
    id_fondo = ctx.tax.id_de("fondo_sin_objeto")
    filas, errores, t0 = [], 0, time.time()
    tareas = [(f["id_objeto"], f, int(f["clase_id"])) for _, f in clf.iterrows()]
    tareas += [(f"fondo:{r}#{i}", {"ruta": r, "bbox_x": b[0], "bbox_y": b[1], "bbox_w": b[2], "bbox_h": b[3],
                                    "ancho": None, "alto": None, "split": s, "grupo_split": g,
                                    "fuente": "fondo"}, id_fondo)
               for i, (r, b, s, g) in enumerate(fondos)]
    for i, (id_obj, f, cid) in enumerate(tareas, 1):
        nombre = _nombre_seguro(id_obj, ".jpg")
        destino = base / f["split"] / f"{cid:02d}" / nombre
        if not destino.exists():
            try:
                im = _redimensionar(_recortar(leer_imagen_pil(f["ruta"]), f,
                                              rc["margen"] if cid != id_fondo else 0.0), rc["lado"])
                destino.parent.mkdir(parents=True, exist_ok=True)
                im.save(destino, quality=rc["calidad_jpeg"])
            except Exception as exc:  # Imagen corrupta: se informa y se sigue
                errores += 1
                if errores <= 5:
                    log(f"⚠️  {id_obj}: {exc}")
                continue
        filas.append({"recorte": destino.relative_to(base).as_posix(), "clase_id": cid,
                      "split": f["split"], "fuente": f["fuente"], "grupo_split": f["grupo_split"]})
        if i % 2000 == 0:
            log(f"  {i:,}/{len(tareas):,} recortes · {i / (time.time() - t0):.0f}/s")
    indice = pd.DataFrame(filas)
    indice.to_csv(base / "indice.csv", index=False)
    log(f"✔ {len(indice):,} recortes en {base} ({errores} errores)")
    return base / "indice.csv"


def seleccionar_entrenamiento(indice: pd.DataFrame, cfg_bal: dict, semilla: int) -> pd.DataFrame:
    """
    Aplica los topes de balanceo SOLO al conjunto de entrenamiento.

    1. Cada fuente con tope (p. ej. off: 0.30) no supera esa fracción dentro de una clase.
    2. Ninguna clase supera `tope_por_clase` recortes.
    """
    ent = indice[indice["split"] == "entrenamiento"]
    partes = []
    for _cid, g in ent.groupby("clase_id"):
        n_total = len(g)
        for fuente, gf in g.groupby("fuente"):
            frac = cfg_bal.get("tope_fraccion_fuente", {}).get(fuente)
            if frac is not None:
                otras = n_total - len(gf)
                maximo = max(1, int(frac / (1 - frac) * otras)) if otras else len(gf)
                gf = gf.sample(min(len(gf), maximo), random_state=semilla)
            partes.append(gf)
    ent = pd.concat(partes) if partes else ent
    tope = cfg_bal.get("tope_por_clase")
    if tope:
        ent = ent.sample(frac=1.0, random_state=semilla).groupby("clase_id").head(tope)
    return ent.reset_index(drop=True)


def pesos_muestreo(clases: np.ndarray, modo: str) -> np.ndarray:
    """
    >>> pesos_muestreo(np.array([0, 0, 0, 0, 1]), "raiz_inversa").round(2).tolist()
    [0.5, 0.5, 0.5, 0.5, 1.0]
    """
    conteo = Counter(clases.tolist())
    if modo == "uniforme":
        return np.ones(len(clases))
    exp = 1.0 if modo == "inversa" else 0.5
    return np.array([1.0 / conteo[c] ** exp for c in clases])


class DatasetRecortes:
    """Dataset de PyTorch sobre el índice de recortes (las imágenes se leen de disco)."""

    def __init__(self, base: Path, filas: pd.DataFrame, transformacion) -> None:
        self.base = base
        self.rutas = filas["recorte"].tolist()
        self.y = filas["clase_id"].astype(int).to_numpy()
        self.t = transformacion

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, i: int):
        from PIL import Image
        with Image.open(self.base / self.rutas[i]) as im:
            return self.t(im.convert("RGB")), int(self.y[i])


def transformaciones(cfg: dict, entrenamiento: bool):
    from torchvision.transforms import v2 as T
    import torch
    tam = cfg["entrenamiento"]["imgsz"]
    if not entrenamiento:
        return T.Compose([T.ToImage(), T.Resize(int(tam * 1.14)), T.CenterCrop(tam),
                          T.ToDtype(torch.float32, scale=True), T.Normalize(MEDIA, DESV)])
    a = cfg["aumentos"]
    j = a["jitter"]
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(tam, scale=tuple(a["escala_recorte"]), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomRotation(a["rotacion"])], p=0.5),
        T.RandomPerspective(distortion_scale=0.3, p=a["perspectiva_p"]),
        T.RandomApply([T.ElasticTransform(alpha=40.0, sigma=6.0)], p=a["elastico_p"]),
        T.ColorJitter(j["brillo"], j["contraste"], j["saturacion"], j["tono"]),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.8))], p=a["desenfoque_p"]),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEDIA, DESV),
        T.RandomErasing(p=a["borrado_p"], scale=(0.02, 0.15)),
    ])


def construir_modelo(cfg: dict, n_clases: int, cargar_imagenet: bool):
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
    if cfg["arquitectura"] != "mobilenet_v3_large":
        raise ValueError(f"Arquitectura no soportada: {cfg['arquitectura']}")
    pesos = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if cargar_imagenet else None
    modelo = mobilenet_v3_large(weights=pesos)
    modelo.classifier[3] = nn.Linear(modelo.classifier[3].in_features, n_clases)
    for p in modelo.features.parameters():
        p.requires_grad = not cfg.get("congelar_backbone", False)
    return modelo


def _exactitud_balanceada(y: np.ndarray, pred: np.ndarray) -> float:
    recalls = [float((pred[y == c] == c).mean()) for c in np.unique(y)]
    return float(np.mean(recalls)) if recalls else 0.0


def _predecir(modelo, cargador, dispositivo) -> tuple[np.ndarray, np.ndarray]:
    import torch
    modelo.eval()
    logits, ys = [], []
    with torch.no_grad():
        for x, y in cargador:
            logits.append(modelo(x.to(dispositivo)).float().cpu())
            ys.append(y)
    if not logits:
        return np.zeros((0, 0), np.float32), np.zeros(0, int)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def etapa_entrenar_clasificador(ctx: Contexto) -> Path | None:
    cfg = ctx.cfg["clasificador"]
    ce = cfg["entrenamiento"]
    ruta_indice = ctx.dir_recortes / "indice.csv"
    if ctx.dry_run:
        log(f"Entrenaría {cfg['arquitectura']} (16 clases, congelar_backbone="
            f"{cfg.get('congelar_backbone', False)}) · {ce}")
        return None
    if not ruta_indice.exists():
        sys.exit("❌ Falta el índice de recortes: ejecute la etapa exportar-recortes.")
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision.transforms import v2 as T

    semilla = ctx.cfg.get("semilla", 42)
    torch.manual_seed(semilla)
    indice = pd.read_csv(ruta_indice)
    ent = seleccionar_entrenamiento(indice, cfg["balanceo"], semilla)
    val = indice[indice["split"] == "validacion"]
    if ent.empty or val.empty:
        sys.exit("❌ Se necesitan recortes de entrenamiento y de validación.")
    n_clases = len(ctx.tax.clases)
    log("Entrenamiento por clase: " + ", ".join(
        f"{ctx.tax.nombres[c]}={n}" for c, n in sorted(Counter(ent["clase_id"]).items())))
    faltan = sorted(set(range(n_clases)) - set(ent["clase_id"]))
    if faltan:
        log(f"⚠️  Clases SIN ejemplos de entrenamiento: {[ctx.tax.nombres[c] for c in faltan]}")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_ent = DatasetRecortes(ctx.dir_recortes, ent, transformaciones(cfg, True))
    ds_val = DatasetRecortes(ctx.dir_recortes, val, transformaciones(cfg, False))
    pesos = torch.as_tensor(pesos_muestreo(ds_ent.y, cfg["balanceo"]["sampler"]), dtype=torch.double)
    comun = dict(batch_size=ce["batch"], num_workers=ce["workers"], pin_memory=dispositivo.type == "cuda",
                 persistent_workers=ce["workers"] > 0)
    cargador_ent = DataLoader(ds_ent, sampler=WeightedRandomSampler(pesos, len(ds_ent), replacement=True),
                              drop_last=len(ds_ent) > ce["batch"], **comun)
    cargador_val = DataLoader(ds_val, shuffle=False, **comun)

    modelo = construir_modelo(cfg, n_clases, cfg.get("preentrenado", True)).to(dispositivo)
    params_backbone = [p for p in modelo.features.parameters() if p.requires_grad]
    params_cabeza = list(modelo.classifier.parameters())
    opt = torch.optim.AdamW([
        {"params": params_backbone, "lr": ce["lr"] * ce["lr_backbone_factor"]},
        {"params": params_cabeza, "lr": ce["lr"]},
    ], weight_decay=ce["weight_decay"])
    pasos_epoca = max(1, len(cargador_ent))
    total, calentamiento = ce["epochs"] * pasos_epoca, ce["warmup_epochs"] * pasos_epoca
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, calentamiento)
                                              if s < calentamiento else
                                              0.5 * (1 + math.cos(math.pi * (s - calentamiento)
                                                                  / max(1, total - calentamiento))))
    criterio = torch.nn.CrossEntropyLoss(label_smoothing=ce["label_smoothing"])
    usar_amp = bool(ce.get("amp")) and dispositivo.type == "cuda"
    escalador = torch.amp.GradScaler("cuda", enabled=usar_amp)
    mezcla = T.RandomChoice([T.CutMix(num_classes=n_clases), T.MixUp(num_classes=n_clases)])

    salida = ctx.dir_clasificador
    salida.mkdir(parents=True, exist_ok=True)
    mejor, sin_mejora, historial = -1.0, 0, []
    for epoca in range(1, ce["epochs"] + 1):
        modelo.train()
        t0, perdida_acum, n = time.time(), 0.0, 0
        for x, y in cargador_ent:
            x, y = x.to(dispositivo, non_blocking=True), y.to(dispositivo, non_blocking=True)
            if random.random() < ce["cutmix_mixup_p"] and len(y) > 1:
                x, y = mezcla(x, y)          # y pasa a ser blanda (one-hot mezclado)
            with torch.autocast(device_type=dispositivo.type, enabled=usar_amp):
                perdida = criterio(modelo(x), y)
            opt.zero_grad(set_to_none=True)
            escalador.scale(perdida).backward()
            escalador.step(opt)
            escalador.update()
            sched.step()
            perdida_acum += float(perdida) * len(x)
            n += len(x)
        logits, yv = _predecir(modelo, cargador_val, dispositivo)
        bal = _exactitud_balanceada(yv, logits.argmax(1))
        historial.append({"epoca": epoca, "perdida": perdida_acum / max(1, n), "val_bal": bal,
                          "segundos": time.time() - t0})
        log(f"Época {epoca}/{ce['epochs']} · pérdida {perdida_acum / max(1, n):.3f} · "
            f"val exactitud balanceada {bal:.1%} · {time.time() - t0:.0f} s")
        if bal > mejor:
            mejor, sin_mejora = bal, 0
            torch.save({"estado": modelo.state_dict(), "clases": ctx.tax.nombres, "val_bal": bal,
                        "epoca": epoca, "cfg": cfg}, salida / "mejor.pt")
        else:
            sin_mejora += 1
            if sin_mejora >= ce["paciencia"]:
                log("Parada temprana: la validación dejó de mejorar.")
                break
    pd.DataFrame(historial).to_csv(salida / "historial.csv", index=False)
    log(f"✔ Mejor exactitud balanceada en validación: {mejor:.1%} -> {salida / 'mejor.pt'}")
    return salida / "mejor.pt"


def _cargar_mejor(ctx: Contexto):
    import torch
    ruta = ctx.dir_clasificador / "mejor.pt"
    if not ruta.exists():
        sys.exit("❌ Falta el clasificador entrenado: ejecute la etapa entrenar-clasificador.")
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    modelo = construir_modelo(ctx.cfg["clasificador"], len(ck["clases"]), cargar_imagenet=False)
    modelo.load_state_dict(ck["estado"])
    return modelo.eval(), ck


def _cargador(ctx: Contexto, split: str):
    from torch.utils.data import DataLoader
    indice = pd.read_csv(ctx.dir_recortes / "indice.csv")
    filas = indice[indice["split"] == split]
    ds = DatasetRecortes(ctx.dir_recortes, filas, transformaciones(ctx.cfg["clasificador"], False))
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0), filas


# ===========================================================================
# 4. CALIBRACIÓN, EXPORTACIÓN Y EVALUACIÓN
# ===========================================================================
def ajustar_temperatura(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperatura T que minimiza la NLL de softmax(logits / T) en validación."""
    import torch
    if len(y) == 0:
        return 1.0
    z, t = torch.as_tensor(logits, dtype=torch.float32), torch.as_tensor(y)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def paso():
        opt.zero_grad()
        perdida = torch.nn.functional.cross_entropy(z / log_t.exp(), t)
        perdida.backward()
        return perdida

    opt.step(paso)
    return float(log_t.exp().clamp(0.05, 20.0))


def etapa_calibrar(ctx: Contexto) -> float:
    if ctx.dry_run:
        log("Calibraría la temperatura sobre validación.")
        return 1.0
    modelo, _ = _cargar_mejor(ctx)
    cargador, _ = _cargador(ctx, "validacion")
    import torch
    logits, y = _predecir(modelo, cargador, torch.device("cpu"))
    T = ajustar_temperatura(logits, y)
    (ctx.dir_clasificador / "temperatura.json").write_text(json.dumps({"temperatura": T}), encoding="utf-8")
    log(f"✔ Temperatura = {T:.3f} (T > 1: el modelo estaba sobreconfiado)")
    return T


def etapa_exportar_clasificador(ctx: Contexto) -> Path | None:
    cfg = ctx.cfg["clasificador"]
    if ctx.dry_run:
        log(f"Exportaría {cfg['exportar']['salida']} (ONNX opset {cfg['exportar']['opset']}).")
        return None
    import torch
    modelo, ck = _cargar_mejor(ctx)
    ruta_t = ctx.dir_clasificador / "temperatura.json"
    T = json.loads(ruta_t.read_text())["temperatura"] if ruta_t.exists() else 1.0

    class ConTemperatura(torch.nn.Module):
        """Exporta logits / T: softmax de la salida = probabilidad calibrada."""

        def __init__(self, red, temperatura: float) -> None:
            super().__init__()
            self.red = red
            self.register_buffer("t", torch.tensor(float(temperatura)))

        def forward(self, x):
            return self.red(x) / self.t

    tam = cfg["entrenamiento"]["imgsz"]
    ctx.modelos.mkdir(parents=True, exist_ok=True)
    destino = ctx.modelos / cfg["exportar"]["salida"]
    envoltura = ConTemperatura(modelo, T).eval()
    x = torch.randn(1, 3, tam, tam)
    argumentos = dict(input_names=["imagen"], output_names=["logits"],
                      dynamic_axes={"imagen": {0: "lote"}, "logits": {0: "lote"}},
                      opset_version=cfg["exportar"]["opset"])
    try:   # torch >= 2.5: exportador clásico explícito (más compatible con cv2.dnn)
        torch.onnx.export(envoltura, x, str(destino), dynamo=False, **argumentos)
    except TypeError:  # torch 2.3-2.4 no conoce el parámetro `dynamo`
        torch.onnx.export(envoltura, x, str(destino), **argumentos)
    with torch.no_grad():
        ref = envoltura(x).numpy()
    try:
        import onnxruntime as ort
        out = ort.InferenceSession(str(destino), providers=["CPUExecutionProvider"]).run(
            None, {"imagen": x.numpy()})[0]
        log(f"  onnxruntime: diferencia máx. {np.abs(out - ref).max():.2e}")
    except ImportError:
        log("  (onnxruntime no instalado: se omite la verificación)")
    try:
        import cv2
        red = cv2.dnn.readNetFromONNX(str(destino))
        red.setInput(x.numpy())
        log(f"  cv2.dnn:     diferencia máx. {np.abs(red.forward() - ref).max():.2e}")
    except Exception as exc:  # cv2.dnn no soporta algunos operadores en versiones viejas
        log(f"  ⚠️  cv2.dnn no pudo cargar el modelo ({exc}); use onnxruntime en la app.")
    meta = {"arquitectura": cfg["arquitectura"], "clases": ck["clases"], "temperatura": T,
            "imgsz": tam, "media": MEDIA, "desv": DESV, "entrada": "RGB NCHW float32 normalizado",
            "taxonomia": "taxonomia/master.yaml v1", "val_exactitud_balanceada": ck["val_bal"],
            "entrenado": time.strftime("%Y-%m-%d %H:%M")}
    destino.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"✔ Clasificador exportado: {destino}")
    return destino


def etapa_evaluar(ctx: Contexto) -> dict | None:
    ev = ctx.cfg["evaluacion"]
    if ctx.dry_run:
        log(f"Evaluaría en prueba con criterios {ev['aceptar_si']}.")
        return None
    import torch
    modelo, _ = _cargar_mejor(ctx)
    cargador, filas = _cargador(ctx, "prueba")
    if len(filas) == 0:
        log("⚠️  No hay recortes de prueba: agregue fotos propias en una sesión 'prueba_*'.")
        return None
    ruta_t = ctx.dir_clasificador / "temperatura.json"
    T = json.loads(ruta_t.read_text())["temperatura"] if ruta_t.exists() else 1.0
    logits, y = _predecir(modelo, cargador, torch.device("cpu"))
    z = logits / T
    probs = np.exp(z - z.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    nombres = ctx.tax.nombres
    pred_caneca, real_caneca = [], []
    for p, yy in zip(probs, y):
        d = ctx.tax.decidir(dict(zip(nombres, p.tolist())))
        pred_caneca.append(d.caneca)
        real_caneca.append(ctx.tax.caneca(nombres[yy]))
    pred_caneca, real_caneca = np.array(pred_caneca, object), np.array(real_caneca, object)
    a_blanca = pred_caneca == "Blanca"
    contaminacion = float((real_caneca[a_blanca] != "Blanca").mean()) if a_blanca.any() else 0.0
    pred = probs.argmax(1)
    recall = {nombres[c]: float((pred[y == c] == c).mean()) for c in np.unique(y)}
    res = {"n": int(len(y)), "exactitud": float((pred == y).mean()),
           "exactitud_balanceada": _exactitud_balanceada(y, pred),
           "caneca_correcta": float((pred_caneca == real_caneca).mean()),
           "contaminacion_blanca": contaminacion, "recall_por_clase": recall,
           "por_fuente": {f: float((pred[(filas["fuente"] == f).to_numpy()] ==
                                     y[(filas["fuente"] == f).to_numpy()]).mean())
                          for f in filas["fuente"].unique()}}
    crit = ev["aceptar_si"]
    criticas = {c: recall.get(c) for c in ev["clases_criticas"]}
    motivos = []
    for c, r in criticas.items():
        if r is None:
            motivos.append(f"⚠️  {c}: sin ejemplos de prueba")
        elif r < crit["recall_minimo_criticas"]:
            motivos.append(f"❌ {c}: recall {r:.1%} < {crit['recall_minimo_criticas']:.0%}")
    if contaminacion > crit["contaminacion_blanca_maxima"]:
        motivos.append(f"❌ contaminación de la Blanca {contaminacion:.1%} > "
                       f"{crit['contaminacion_blanca_maxima']:.0%}")
    if res["exactitud_balanceada"] < crit["exactitud_balanceada_minima"]:
        motivos.append(f"❌ exactitud balanceada {res['exactitud_balanceada']:.1%} < "
                       f"{crit['exactitud_balanceada_minima']:.0%}")
    res["aceptado"] = not any(m.startswith("❌") for m in motivos)
    res["motivos"] = motivos
    lineas = [f"Evaluación en PRUEBA · {res['n']} recortes",
              f"  Exactitud:              {res['exactitud']:.1%}",
              f"  Exactitud balanceada:   {res['exactitud_balanceada']:.1%}",
              f"  Caneca correcta:        {res['caneca_correcta']:.1%}",
              f"  Contaminación Blanca:   {contaminacion:.1%}", "  Recall por clase:"]
    lineas += [f"    {k:26s} {v:6.1%}" for k, v in sorted(recall.items())]
    lineas += ["  Por fuente:"] + [f"    {k:10s} {v:6.1%}" for k, v in res["por_fuente"].items()]
    lineas += [""] + (motivos or ["✅ Cumple todos los criterios"])
    lineas.append("=> MODELO " + ("ACEPTADO" if res["aceptado"] else "NO ACEPTADO"))
    texto = "\n".join(lineas)
    print(texto)
    (ctx.modelos / "reporte_pipeline.txt").write_text(texto, encoding="utf-8")
    (ctx.dir_clasificador / "evaluacion.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")
    return res


# ===========================================================================
# 5. ORQUESTACIÓN
# ===========================================================================
FUNCIONES = {
    "preparar-pesos": etapa_preparar_pesos,
    "exportar-detector": etapa_exportar_detector,
    "entrenar-detector": etapa_entrenar_detector,
    "exportar-recortes": etapa_exportar_recortes,
    "entrenar-clasificador": etapa_entrenar_clasificador,
    "calibrar": etapa_calibrar,
    "exportar-clasificador": etapa_exportar_clasificador,
    "evaluar": etapa_evaluar,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=RAIZ / "entrenamiento" / "pipeline.yaml")
    p.add_argument("--etapas", default="todo",
                   help=f"Lista separada por comas o 'todo' ({', '.join(ETAPAS_POR_DEFECTO)}).")
    p.add_argument("--rapido", action="store_true", help="Prueba de humo: 1 época, pocas imágenes.")
    p.add_argument("--dry-run", action="store_true", help="Muestra el plan y los conteos sin entrenar.")
    p.add_argument("--solo-detector", action="store_true")
    p.add_argument("--solo-clasificador", action="store_true")
    args = p.parse_args(argv)

    etapas = list(ETAPAS_POR_DEFECTO) if args.etapas == "todo" else [e.strip() for e in args.etapas.split(",")]
    desconocidas = [e for e in etapas if e not in FUNCIONES]
    if desconocidas:
        p.error(f"Etapas desconocidas: {desconocidas}. Válidas: {list(ETAPAS)}")
    if args.solo_detector:
        etapas = [e for e in etapas if "detector" in e or e == "preparar-pesos"]
    if args.solo_clasificador:
        etapas = [e for e in etapas if "detector" not in e]
    etapas = [e for e in ETAPAS if e in etapas]  # Siempre en el orden correcto

    ctx = cargar_contexto(args.config, args.rapido, args.dry_run)
    log(f"Pipeline: {' -> '.join(etapas)}" + (" [RÁPIDO]" if args.rapido else "")
        + (" [DRY-RUN]" if args.dry_run else ""))
    for etapa in etapas:
        log(f"━━ {etapa} ━━")
        FUNCIONES[etapa](ctx)
    log("✅ Pipeline terminado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
