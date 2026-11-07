"""
reentrenar.py — Reentrenamiento del clasificador con TACO + TrashNet + fotos propias
====================================================================================
Un solo comando que corrige la «brecha de dominio» del modelo (paquetes metalizados,
icopor, aerosoles) y SOLO reemplaza el modelo de la app si demuestra que mejora.

Qué hace, paso a paso:
    1. RED BASE     Extrae la MobileNetV2 (sin su capa final) directamente de
                    models/trashnet_mobilenetv2.onnx. No hay que descargar ni convertir
                    pesos Caffe: la red ya está dentro del modelo actual.
    2. DATOS        Reúne tres fuentes y las separa SIN mezclar fotos entre conjuntos:
                      * TrashNet: misma partición 70/15/15 (semilla 42) del entrenamiento
                        original -> la comparación con el 89,2 % reportado es justa.
                      * TACO (o cualquier COCO): partición POR LOTES (batch_*). Las fotos
                        de prueba son de lotes que el modelo nunca vio.
                      * Fotos propias (opcional): carpetas por clase (icopor/, metal/...).
                    Las etiquetas salen de annotations.json / nombres de carpeta con el
                    mismo mapeo de `ingesta_publica.py`. No hay que etiquetar a mano.
    3. AUMENTO      Las fotos de entrenamiento se multiplican con albumentations
                    (destellos, brillo, encuadre, ruido, compresión de webcam).
    4. ENTRENAMIENTO Transfer learning: la red base convierte cada foto en 1280 números
                    y se entrena una regresión logística (6 clases TrashNet) con pesos
                    balanceados. El hiperparámetro C se elige con el conjunto de validación.
    5. COMPARACIÓN  Modelo ACTUAL vs. NUEVO en las fotos de prueba, en dos condiciones:
                    «limpia» (foto tal cual) y «webcam» (foto degradada con aumentos).
    6. DECISIÓN     El modelo nuevo reemplaza al actual solo si mejora en TACO/fotos
                    propias y no empeora en TrashNet más que la tolerancia (3 puntos).
                    El modelo anterior se guarda en models/respaldo/<fecha>/.

Requisitos (además de requirements.txt):
    pip install -r entrenamiento/requirements-entrenamiento.txt

Uso (desde la carpeta del proyecto):
    python entrenamiento/reentrenar.py --taco ../TACO/data --trashnet entrenamiento/dataset-resized
    python entrenamiento/reentrenar.py --taco ../TACO/data --fotos fotos_propias --no-reemplazar
    python entrenamiento/reentrenar.py --restaurar models/respaldo/20261107_0100

TrashNet: https://github.com/garythung/trashnet -> data/dataset-resized.zip (descomprimir).
Tiempo estimado en un portátil (CPU): 10-25 min la primera vez; las ejecuciones
siguientes reutilizan la caché de características (entrenamiento/cache/) y tardan ~1-3 min.

Autor: Johan Lasso
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import ingesta_publica as ip  # noqa: E402  (reutiliza lectura COCO, mapeo y aumentos)

# ===========================================================================
# COMPATIBILIDAD CON INGESTA_PUBLICA
# ===========================================================================
@dataclass(frozen=True)
class ClaseNormativa:
    nombre: str
    contenedor: str | None = None
    evaluable: bool = True
    metalizada: bool = False
    critica: bool = False
    clase_trashnet: str | None = None

@dataclass
class MuestraCompat:
    archivo: str
    ruta: Path | None = None
    bbox: tuple | None = None
    tam_anotado: tuple | None = None
    clase: ClaseNormativa | None = None
    detalle: str = ""
    critica: bool = False

ip.ClaseNormativa = ClaseNormativa
ip.Muestra = MuestraCompat

MAPEO_TRASHNET_A_CLASE = {
    "cardboard": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="cardboard"),
    "glass": ClaseNormativa("Vidrio", "Blanco", clase_trashnet="glass"),
    "metal": ClaseNormativa("Metal", "Blanco", clase_trashnet="metal"),
    "paper": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="paper"),
    "plastic": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "trash": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
}

MAPEO_TACO = {
    "Aluminium foil": ClaseNormativa("Metal", "Blanco", clase_trashnet="metal", metalizada=True),
    "Battery": ClaseNormativa("Peligroso", None, evaluable=False, clase_trashnet=None),
    "Aluminium blister pack": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash", metalizada=True),
    "Carded blister pack": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Other plastic bottle": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "Clear plastic bottle": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "Glass bottle": ClaseNormativa("Vidrio", "Blanco", clase_trashnet="glass"),
    "Plastic bottle cap": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "Metal bottle cap": ClaseNormativa("Metal", "Blanco", clase_trashnet="metal"),
    "Broken glass": ClaseNormativa("Vidrio", "Blanco", clase_trashnet="glass"),
    "Food Can": ClaseNormativa("Metal", "Blanco", clase_trashnet="metal"),
    "Aerosol": ClaseNormativa("Metal", "Blanco", clase_trashnet="metal", critica=True),
    "Drink can": ClaseNormativa("Metal", "Blanco", clase_trashnet="metal"),
    "Toilet tube": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="cardboard"),
    "Other carton": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="cardboard"),
    "Egg carton": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="cardboard"),
    "Drink carton": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="cardboard"),
    "Corrugated carton": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="cardboard"),
    "Meal carton": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="cardboard"),
    "Pizza box": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="cardboard"),
    "Paper cup": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Disposable plastic cup": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "Foam cup": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash", critica=True),
    "Glass cup": ClaseNormativa("Vidrio", "Blanco", clase_trashnet="glass"),
    "Other plastic cup": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "Food waste": ClaseNormativa("Orgánico", "Verde", evaluable=False, clase_trashnet=None),
    "Glass jar": ClaseNormativa("Vidrio", "Blanco", clase_trashnet="glass"),
    "Plastic lid": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "Metal lid": ClaseNormativa("Metal", "Blanco", clase_trashnet="metal"),
    "Other plastic": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "Magazine paper": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="paper"),
    "Tissues": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Wrapping paper": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="paper"),
    "Normal paper": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="paper"),
    "Paper bag": ClaseNormativa("Papel/Cartón", "Blanco", clase_trashnet="paper"),
    "Plastified paper bag": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Plastic film": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Six pack rings": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Garbage bag": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Other plastic wrapper": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Single-use carrier bag": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Polypropylene bag": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Crisp packet": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash", metalizada=True, critica=True),
    "Spread tub": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "Tupperware": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "Disposable food container": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "Foam food container": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash", critica=True),
    "Other plastic container": ClaseNormativa("Plástico", "Blanco", clase_trashnet="plastic"),
    "Plastic glooves": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Plastic utensils": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Pop tab": ClaseNormativa("Metal", "Blanco", clase_trashnet="metal"),
    "Rope & strings": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Scrap metal": ClaseNormativa("Metal", "Blanco", clase_trashnet="metal"),
    "Shoe": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Squeezable tube": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Plastic straw": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Paper straw": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
    "Styrofoam piece": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash", critica=True),
    "Unlabeled litter": ClaseNormativa("No identificado", None, evaluable=False, clase_trashnet=None),
    "Cigarette": ClaseNormativa("No aprovechable", "Negro", clase_trashnet="trash"),
}

def _mapear_cat(nombre: str, supercategoria: str | None = None) -> ClaseNormativa:
    if nombre in MAPEO_TRASHNET_A_CLASE:
        return MAPEO_TRASHNET_A_CLASE[nombre]
    if nombre in MAPEO_TACO:
        return MAPEO_TACO[nombre]
    for k, v in MAPEO_TACO.items():
        if k.lower() == nombre.lower():
            return v
    for k, v in MAPEO_TRASHNET_A_CLASE.items():
        if k.lower() == nombre.lower():
            return v
    if supercategoria:
        for k, v in MAPEO_TACO.items():
            if k.lower() == supercategoria.lower():
                return v
    return ClaseNormativa(nombre, None, evaluable=False, clase_trashnet=None)

ip.mapear_categoria = _mapear_cat

def _buscar_anotaciones(ruta: Path) -> list[Path]:
    if (ruta / "annotations.json").is_file():
        return [ruta / "annotations.json"]
    jsons = list(ruta.glob("**/annotations.json")) + list(ruta.glob("**/*_annotations.coco.json"))
    return sorted(set(jsons))

ip.buscar_anotaciones = _buscar_anotaciones

def _muestras_coco(rutas_json: list[Path], ruta_raiz: Path, lado_minimo: int = 48,
                   max_por_imagen: int = 3, incluir_sin_mapeo: bool = False):
    muestras = []
    faltantes = 0
    for r_json in rutas_json:
        dir_base = r_json.parent
        with open(r_json, encoding="utf-8") as f:
            datos = json.load(f)
        imgs = {im["id"]: im for im in datos.get("images", [])}
        cats = {c["id"]: c["name"] for c in datos.get("categories", [])}
        por_imagen = defaultdict(list)
        for an in datos.get("annotations", []):
            if "bbox" not in an or len(an["bbox"]) < 4:
                continue
            _, _, w, h = an["bbox"]
            if min(w, h) >= lado_minimo:
                por_imagen[an["image_id"]].append(an)
        for id_img, anotaciones in por_imagen.items():
            im_info = imgs.get(id_img)
            if not im_info:
                continue
            arch_img = dir_base / im_info["file_name"]
            if not arch_img.is_file():
                arch_img = ruta_raiz / im_info["file_name"]
            if not arch_img.is_file():
                faltantes += 1
                continue
            anotaciones = sorted(anotaciones, key=lambda a: a.get("area", a["bbox"][2]*a["bbox"][3]), reverse=True)[:max_por_imagen]
            for an in anotaciones:
                nom_cat = cats.get(an["category_id"], "")
                cn = _mapear_cat(nom_cat)
                if not incluir_sin_mapeo and cn.clase_trashnet is None:
                    continue
                tam_anotado = (im_info.get("height"), im_info.get("width"))
                rel_path = im_info["file_name"].replace("\\", "/")
                muestras.append(MuestraCompat(
                    archivo=f"{rel_path}#{an['id']}",
                    ruta=arch_img,
                    bbox=tuple(an["bbox"]),
                    tam_anotado=tam_anotado,
                    clase=cn,
                    detalle=nom_cat,
                    critica=getattr(cn, "critica", False)
                ))
    return muestras, faltantes

ip.muestras_coco = _muestras_coco

def _muestras_carpeta(ruta: Path, solo_prueba: bool = False):
    muestras = []
    if ruta.is_dir():
        for sub in ruta.iterdir():
            if not sub.is_dir():
                continue
            cn = _mapear_cat(sub.name)
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                for f in sorted(sub.glob(f"*{ext}")):
                    muestras.append(MuestraCompat(
                        archivo=f"{sub.name}/{f.name}",
                        ruta=f,
                        bbox=None,
                        tam_anotado=None,
                        clase=cn,
                        detalle=sub.name
                    ))
    return muestras

ip.muestras_carpeta = _muestras_carpeta

def _recortar_func(imagen: np.ndarray, bbox: tuple[float, float, float, float] | None, margen: float = 0.30) -> np.ndarray:
    if bbox is None:
        return imagen
    h_img, w_img = imagen.shape[:2]
    x, y, w, h = bbox
    pad_x = int(w * margen)
    pad_y = int(h * margen)
    x0 = max(0, int(x - pad_x))
    y0 = max(0, int(y - pad_y))
    x1 = min(w_img, int(x + w + pad_x))
    y1 = min(h_img, int(y + h + pad_y))
    rec = imagen[y0:y1, x0:x1]
    return rec if rec.size > 0 else imagen

ip.recortar = _recortar_func
_orig_leer = ip.leer_imagen
ip.leer_imagen = lambda ruta, tam_anotado=None: _orig_leer(Path(ruta))
_orig_sim = ip.simular_camara
ip.simular_camara = lambda bgr, calidad_jpeg=75: _orig_sim(bgr, calidad_jpeg)

def _generar_variantes(muestra: MuestraCompat, base: np.ndarray, aumentos: int = 2,
                       incluir_original: bool = True, semilla: int = 42, calidad_jpeg: int = 75):
    if incluir_original:
        yield 0, ip.simular_camara(base, calidad_jpeg)
    try:
        import albumentations as A
        alto, ancho = base.shape[:2]
        radio = max(8, int(0.30 * min(alto, ancho)))
        metalizada = getattr(muestra.clase, "metalizada", False) or muestra.critica
        aug = A.Compose([
            A.LongestMaxSize(max_size=640, p=1.0),
            A.RandomCrop(height=max(16, int(alto * 0.85)), width=max(16, int(ancho * 0.85)), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.20), contrast_limit=(-0.20, 0.20), p=0.8),
            A.ImageCompression(compression_type="jpeg", quality_range=(55, 85), p=0.9),
        ], seed=semilla)
        for i in range(1, aumentos + 1):
            yield i, aug(image=base)["image"]
    except Exception:
        rng = np.random.default_rng(semilla)
        for i in range(1, aumentos + 1):
            factor = rng.uniform(0.7, 1.3)
            yield i, np.clip(base.astype(np.float32) * factor, 0, 255).astype(np.uint8)

ip.generar_variantes = _generar_variantes

try:
    import onnx
    from onnx import helper, numpy_helper
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split
except ImportError as _exc:  # pragma: no cover - depende del entorno
    sys.exit(f"❌ Falta {_exc.name}. Ejecute: pip install -r entrenamiento/requirements-entrenamiento.txt")

CLASES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]   # Orden del modelo
GRUPO_CLASE = {"cardboard": "Papel/Cartón", "paper": "Papel/Cartón", "glass": "Vidrio",
               "metal": "Metal", "plastic": "Plástico", "trash": "No aprovechable"}
GRUPOS = sorted(set(GRUPO_CLASE.values()))
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESV = np.array([0.229, 0.224, 0.225], np.float32)

DIR_MODELOS = RAIZ / "models"
MODELO_APP = DIR_MODELOS / "trashnet_mobilenetv2.onnx"
DIR_CACHE = RAIZ / "entrenamiento" / "cache"
DIR_SALIDAS = RAIZ / "entrenamiento" / "salidas"
CONJUNTOS = ("entrenamiento", "validacion", "prueba")


# ===========================================================================
# 1. RED BASE (backbone) A PARTIR DEL MODELO ACTUAL
# ===========================================================================
@dataclass
class Cabeza:
    """Capa final lineal del modelo: logits = f · Wᵀ + b."""

    W: np.ndarray            # (6, 1280)
    b: np.ndarray            # (6,)
    nombre_W: str
    nombre_b: str
    entrada: str             # Tensor de características (1280) que alimenta la capa

    def logits(self, X: np.ndarray) -> np.ndarray:
        """Aplica la capa a una matriz de características (N, 1280)."""
        return X @ self.W.T + self.b


def leer_cabeza(modelo: onnx.ModelProto) -> Cabeza:
    """
    Localiza la capa final (nodo Gemm que produce la salida del grafo).

    Args:
        modelo: Modelo ONNX completo (red base + capa final).

    Returns:
        Pesos y nombres de la capa final.

    Raises:
        ValueError: Si el modelo no termina en una capa Gemm.
    """
    g = modelo.graph
    salida = g.output[0].name
    nodo = next((n for n in g.node if salida in n.output), None)
    if nodo is None or nodo.op_type != "Gemm":
        raise ValueError("El modelo no termina en una capa Gemm; no se puede separar la red base.")
    inits = {i.name: i for i in g.initializer}
    W = numpy_helper.to_array(inits[nodo.input[1]]).astype(np.float32)
    b = numpy_helper.to_array(inits[nodo.input[2]]).astype(np.float32)
    trans_b = next((a.i for a in nodo.attribute if a.name == "transB"), 0)
    if not trans_b:
        W = W.T
    if W.shape[0] != len(CLASES):
        raise ValueError(f"La capa final tiene {W.shape[0]} salidas; se esperaban {len(CLASES)}.")
    return Cabeza(W, b, nodo.input[1], nodo.input[2], nodo.input[0])


def extraer_backbone(ruta_modelo: Path, ruta_salida: Path) -> tuple[Path, Cabeza]:
    """
    Crea la red base (sin capa final) con lote dinámico para extraer características.

    Args:
        ruta_modelo: Modelo actual de la app.
        ruta_salida: Dónde guardar el backbone.

    Returns:
        (ruta del backbone, capa final del modelo actual).
    """
    modelo = onnx.load(str(ruta_modelo))
    cabeza = leer_cabeza(modelo)
    g = modelo.graph
    nodo_final = next(n for n in g.node if g.output[0].name in n.output)
    g.node.remove(nodo_final)
    for nombre in (cabeza.nombre_W, cabeza.nombre_b):
        g.initializer.remove(next(i for i in g.initializer if i.name == nombre))
    del g.output[:]
    g.output.append(helper.make_tensor_value_info(cabeza.entrada, onnx.TensorProto.FLOAT,
                                                  ["N", cabeza.W.shape[1]]))
    g.input[0].type.tensor_type.shape.dim[0].dim_param = "N"   # Lote dinámico
    onnx.checker.check_model(modelo)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(modelo, str(ruta_salida))
    return ruta_salida, cabeza


def preprocesar(imagenes: list[np.ndarray]) -> np.ndarray:
    """
    Preprocesamiento IDÉNTICO a `vision_engine.ClasificadorResiduos._inferir_red`.

    Args:
        imagenes: Lista de imágenes BGR.

    Returns:
        Tensor NCHW float32 normalizado con media/desviación de ImageNet.
    """
    lote = []
    for bgr in imagenes:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        lote.append(((rgb - MEDIA) / DESV).transpose(2, 0, 1))
    return np.stack(lote).astype(np.float32)


class Extractor:
    """Ejecuta la red base con cv2.dnn (sin TensorFlow/PyTorch) por lotes."""

    def __init__(self, ruta_backbone: Path, tam_lote: int = 32) -> None:
        """Carga el backbone y verifica si acepta lotes > 1."""
        self.red = cv2.dnn.readNetFromONNX(str(ruta_backbone))
        self.tam_lote = tam_lote
        try:
            self._forward(np.zeros((2, 3, 224, 224), np.float32))
        except cv2.error:
            self.tam_lote = 1

    def _forward(self, blob: np.ndarray) -> np.ndarray:
        self.red.setInput(blob)
        return self.red.forward().reshape(blob.shape[0], -1)

    def caracteristicas(self, imagenes: list[np.ndarray]) -> np.ndarray:
        """
        Convierte imágenes BGR en vectores de 1280 características.

        Args:
            imagenes: Lista de imágenes BGR.

        Returns:
            Matriz (N, 1280) float32.
        """
        partes = [self._forward(preprocesar(imagenes[i:i + self.tam_lote]))
                  for i in range(0, len(imagenes), self.tam_lote)]
        return np.vstack(partes).astype(np.float32)


# ===========================================================================
# 2. DATOS: TrashNet, TACO/COCO y fotos propias, separados sin fugas
# ===========================================================================
@dataclass(frozen=True)
class Ejemplo:
    """Una foto/objeto con su clase TrashNet y el conjunto al que pertenece."""

    muestra: ip.Muestra
    y: int
    conjunto: str            # entrenamiento / validacion / prueba
    fuente: str              # trashnet / taco / fotos


def _clase_trashnet(m: ip.Muestra) -> int | None:
    """Índice TrashNet de una muestra o None si su clase no tiene equivalente."""
    if m.clase is None or m.clase.clase_trashnet is None:
        return None
    return CLASES.index(m.clase.clase_trashnet)


def cargar_trashnet(ruta: Path) -> list[Ejemplo]:
    """
    Carga TrashNet con la MISMA partición que `entrenar_trashnet.py` (semilla 42).

    Args:
        ruta: Carpeta `dataset-resized` con las 6 subcarpetas.

    Returns:
        Ejemplos etiquetados con su conjunto.
    """
    rutas, y = [], []
    for k, clase in enumerate(CLASES):
        archivos = sorted((ruta / clase).glob("*.jpg"))
        rutas += archivos
        y += [k] * len(archivos)
    if not rutas:
        sys.exit(f"❌ {ruta} no contiene las carpetas de TrashNet ({', '.join(CLASES)}).")
    y = np.array(y)
    idx = np.arange(len(rutas))
    i_ent, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    i_val, i_pru = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=42)
    conjunto = {**{int(i): "entrenamiento" for i in i_ent}, **{int(i): "validacion" for i in i_val},
                **{int(i): "prueba" for i in i_pru}}
    return [Ejemplo(ip.Muestra(archivo=f"trashnet/{rutas[i].parent.name}/{rutas[i].name}",
                               ruta=rutas[i], bbox=None, tam_anotado=None,
                               clase=ip.mapear_categoria(CLASES[y[i]]), detalle=CLASES[y[i]]),
                    int(y[i]), conjunto[i], "trashnet") for i in range(len(rutas))]


def normalizar_lotes(texto: str) -> set[str]:
    """
    Convierte «14,15» en {'batch_14', 'batch_15'}; los nombres se dejan igual («test»).

    >>> sorted(normalizar_lotes("14, 15,test"))
    ['batch_14', 'batch_15', 'test']
    """
    return {f"batch_{t}" if t.isdigit() else t for t in (s.strip() for s in texto.split(",")) if t}


def cargar_coco(ruta: Path, lotes_val: set[str], lotes_prueba: set[str],
                args: argparse.Namespace) -> list[Ejemplo]:
    """
    Carga un dataset COCO (TACO o Roboflow) particionado POR LOTE/CARPETA.

    El lote es la primera carpeta de la ruta relativa de la foto (batch_7 en TACO,
    train/valid/test en Roboflow). Todas las anotaciones de una misma foto quedan en el
    mismo conjunto, así el modelo nunca se evalúa con una foto que ya vio.

    Args:
        ruta: Carpeta del dataset (la que contiene annotations.json).
        lotes_val: Lotes destinados a validación (selección de C).
        lotes_prueba: Lotes destinados a la comparación final.
        args: Argumentos de línea de comandos.

    Returns:
        Ejemplos con clase TrashNet (se omiten orgánico, peligroso y sin etiqueta).
    """
    rutas_json = ip.buscar_anotaciones(ruta)
    if not rutas_json:
        sys.exit(f"❌ No se encontró annotations.json ni *_annotations.coco.json en {ruta}")
    muestras, _ = ip.muestras_coco(rutas_json, ruta, lado_minimo=args.lado_minimo,
                                   max_por_imagen=args.max_por_imagen, incluir_sin_mapeo=False)
    ejemplos, lotes = [], Counter()
    for m in muestras:
        k = _clase_trashnet(m)
        if k is None:
            continue
        lote = m.archivo.split("/", 1)[0]
        lotes[lote] += 1
        conjunto = "prueba" if lote in lotes_prueba else "validacion" if lote in lotes_val \
            else "entrenamiento"
        ejemplos.append(Ejemplo(m, k, conjunto, "taco"))
    faltan = (lotes_val | lotes_prueba) - set(lotes)
    if faltan:
        print(f"⚠️  Lotes inexistentes en el dataset: {sorted(faltan)}. Disponibles: {sorted(lotes)}")
    return ejemplos


def cargar_fotos(ruta: Path) -> list[Ejemplo]:
    """
    Carga fotos propias organizadas en subcarpetas por clase (icopor/, metal/, ...).

    La partición 70/15/15 se decide con un hash estable del nombre de archivo, de modo
    que al agregar fotos nuevas las anteriores no cambian de conjunto.

    Args:
        ruta: Carpeta raíz de las fotos.

    Returns:
        Ejemplos con clase TrashNet.
    """
    ejemplos, sin_clase = [], Counter()
    for m in ip.muestras_carpeta(ruta, solo_prueba=False):
        k = _clase_trashnet(m)
        if k is None:
            sin_clase[m.detalle] += 1
            continue
        h = zlib.crc32(m.archivo.encode()) % 100
        conjunto = "prueba" if h < 15 else "validacion" if h < 30 else "entrenamiento"
        ejemplos.append(Ejemplo(m, k, conjunto, "fotos"))
    if sin_clase:
        print(f"⚠️  Carpetas de fotos sin clase reconocida (renómbrelas): {dict(sin_clase)}")
    return ejemplos


# ===========================================================================
# 3. CARACTERÍSTICAS (con caché en disco)
# ===========================================================================
def _huella(ejemplos: list[Ejemplo], ruta_backbone: Path, args: argparse.Namespace) -> str:
    """Huella de la configuración: cambia si cambian los datos, el modelo o los aumentos."""
    h = hashlib.sha1()
    h.update(hashlib.sha1(ruta_backbone.read_bytes()).digest())
    h.update(f"{args.aumentos}|{args.margen}|{args.semilla}|v1".encode())
    for e in ejemplos:
        h.update(f"{e.muestra.archivo}|{e.conjunto}|{e.y}\n".encode())
    return h.hexdigest()[:16]


def imagenes_de(e: Ejemplo, args: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    """
    Genera las imágenes que se extraen para un ejemplo.

    Entrenamiento: original + `args.aumentos` variantes aumentadas.
    Validación/prueba: «limpia» (TrashNet sin tocar; COCO con degradación de webcam
    determinista) y «webcam» (una variante aumentada con semilla distinta).

    Args:
        e: Ejemplo a procesar.
        args: Argumentos de línea de comandos.

    Returns:
        Lista de (condición, imagen BGR).
    """
    original = ip.leer_imagen(e.muestra.ruta, e.muestra.tam_anotado)
    base = ip.recortar(original, e.muestra.bbox, args.margen)
    if e.conjunto == "entrenamiento":
        return [("limpia" if k == 0 else "aumentada", img) for k, img in ip.generar_variantes(
            e.muestra, base, aumentos=args.aumentos, incluir_original=True,
            semilla=args.semilla, calidad_jpeg=75)]
    limpia = base if e.fuente == "trashnet" else ip.simular_camara(base)
    webcam = next(ip.generar_variantes(e.muestra, base, aumentos=1, incluir_original=False,
                                       semilla=args.semilla + 7919, calidad_jpeg=75))[1]
    return [("limpia", limpia), ("webcam", webcam)]


def construir_caracteristicas(ejemplos: list[Ejemplo], extractor: Extractor, huella: str,
                              args: argparse.Namespace) -> dict[str, np.ndarray]:
    """
    Extrae (o lee de caché) las características de todos los ejemplos.

    Args:
        ejemplos: Ejemplos de todas las fuentes.
        extractor: Red base.
        huella: Identificador de la configuración (nombre del archivo de caché).
        args: Argumentos de línea de comandos.

    Returns:
        Diccionario de arreglos alineados: X, y, conjunto, fuente, condicion,
        detalle, critica, archivo.
    """
    ruta_cache = DIR_CACHE / f"caracteristicas_{huella}.npz"
    if ruta_cache.is_file() and not args.sin_cache:
        print(f"♻️  Usando características en caché: {ruta_cache.name}")
        with np.load(ruta_cache, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}

    cols: dict[str, list] = defaultdict(list)
    X: list[np.ndarray] = []
    pendientes: list[np.ndarray] = []
    errores = 0
    orden = sorted(ejemplos, key=lambda e: str(e.muestra.ruta))   # Aprovecha la caché de fotos
    t0 = time.perf_counter()
    for i, e in enumerate(orden, start=1):
        try:
            variantes = imagenes_de(e, args)
        except Exception as exc:  # Foto corrupta o caja inválida: se omite
            errores += 1
            if args.verbose:
                print(f"\n  ✗ {e.muestra.archivo}: {exc}")
            continue
        for condicion, img in variantes:
            pendientes.append(img)
            cols["y"].append(e.y)
            cols["conjunto"].append(e.conjunto)
            cols["fuente"].append(e.fuente)
            cols["condicion"].append(condicion)
            cols["detalle"].append(e.muestra.detalle)
            cols["critica"].append(e.muestra.critica)
            cols["archivo"].append(e.muestra.archivo)
        if len(pendientes) >= 64:
            X.append(extractor.caracteristicas(pendientes))
            pendientes.clear()
        if i % 25 == 0 or i == len(orden):
            vel = i / max(1e-6, time.perf_counter() - t0)
            print(f"\r  🧮 Características: {i}/{len(orden)} fotos/objetos  {vel:5.1f}/s  "
                  f"~{(len(orden) - i) / max(vel, 1e-6) / 60:4.1f} min restantes ",
                  end="", flush=True)
    if pendientes:
        X.append(extractor.caracteristicas(pendientes))
    print()
    if errores:
        print(f"⚠️  {errores} fotos omitidas por errores de lectura (use -v para el detalle).")
    datos = {"X": np.vstack(X)} | {k: np.array(v) for k, v in cols.items()}
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta_cache, **datos)
    return datos


# ===========================================================================
# 4. ENTRENAMIENTO DE LA CAPA FINAL
# ===========================================================================
def entrenar(datos: dict[str, np.ndarray], grilla_c: list[float]) -> tuple[Cabeza, float, float]:
    """
    Entrena la regresión logística y elige C con la exactitud balanceada de validación.

    La exactitud balanceada (promedio del recall por clase) evita que el modelo
    «gane» prediciendo solo las clases mayoritarias de TACO.

    Args:
        datos: Salida de `construir_caracteristicas`.
        grilla_c: Valores de C a probar.

    Returns:
        (capa final con la estandarización ya plegada, C elegido, exactitud de validación).
    """
    ent = datos["conjunto"] == "entrenamiento"
    val = datos["conjunto"] == "validacion"
    X_ent, y_ent = datos["X"][ent], datos["y"][ent]
    X_val, y_val = datos["X"][val], datos["y"][val]
    faltantes = set(range(len(CLASES))) - set(y_ent.tolist())
    if faltantes:
        sys.exit(f"❌ Sin ejemplos de entrenamiento para: {[CLASES[k] for k in faltantes]}")
    mu, sd = X_ent.mean(0), X_ent.std(0) + 1e-6
    print(f"\n🏋️  Entrenando capa final con {len(y_ent):,} ejemplos "
          f"({', '.join(f'{c}={n}' for c, n in zip(CLASES, np.bincount(y_ent, minlength=6)))})")
    mejor = None
    for C in grilla_c:
        t0 = time.perf_counter()
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit((X_ent - mu) / sd, y_ent)
        acc = balanced_accuracy_score(y_val, clf.predict((X_val - mu) / sd)) if len(y_val) else 0.0
        print(f"   C={C:<5} exactitud balanceada validación = {acc:.3f}  ({time.perf_counter() - t0:.0f}s)")
        if mejor is None or acc > mejor[0]:
            mejor = (acc, C, clf)
    acc, C, clf = mejor
    # Plegado: W·((f - mu)/sd) + b  =  (W/sd)·f + (b - W·mu/sd)
    W = (clf.coef_ / sd).astype(np.float32)
    b = (clf.intercept_ - (clf.coef_ * (mu / sd)).sum(1)).astype(np.float32)
    return Cabeza(W, b, "", "", ""), C, acc


def exportar_modelo(modelo_actual: Path, cabeza: Cabeza, salida: Path) -> Path:
    """
    Copia el modelo actual sustituyendo SOLO los pesos de la capa final.

    El archivo resultante tiene exactamente la misma estructura, nombres y forma de
    entrada que el original, por lo que `vision_engine.py` lo usa sin cambios.

    Args:
        modelo_actual: Modelo de la app.
        cabeza: Nueva capa final.
        salida: Ruta del modelo nuevo.

    Returns:
        Ruta del modelo guardado.
    """
    modelo = onnx.load(str(modelo_actual))
    original = leer_cabeza(modelo)
    trans_b = next(a.i for n in modelo.graph.node if original.nombre_W in n.input
                   for a in n.attribute if a.name == "transB")
    nuevos = {original.nombre_W: cabeza.W if trans_b else cabeza.W.T, original.nombre_b: cabeza.b}
    for init in modelo.graph.initializer:
        if init.name in nuevos:
            init.CopyFrom(numpy_helper.from_array(nuevos[init.name].astype(np.float32), init.name))
    onnx.checker.check_model(modelo)
    salida.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(modelo, str(salida))
    return salida


def verificar_equivalencia(ruta_modelo: Path, cabeza: Cabeza, extractor: Extractor,
                           imagenes: list[np.ndarray]) -> float:
    """
    Comprueba que (backbone + cabeza) == modelo ONNX completo en unas imágenes.

    Args:
        ruta_modelo: Modelo completo a verificar.
        cabeza: Capa final esperada.
        extractor: Red base.
        imagenes: Imágenes de prueba.

    Returns:
        Máxima diferencia absoluta entre logits (debe ser ~1e-4 o menor).
    """
    red = cv2.dnn.readNetFromONNX(str(ruta_modelo))
    esperados = cabeza.logits(extractor.caracteristicas(imagenes))
    dif = 0.0
    for img, esp in zip(imagenes, esperados):
        red.setInput(preprocesar([img]))
        dif = max(dif, float(np.abs(red.forward().flatten() - esp).max()))
    return dif


# ===========================================================================
# 5. EVALUACIÓN Y DECISIÓN
# ===========================================================================
def metricas(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """
    Métricas por clase TrashNet y por grupo del sistema.

    Args:
        y: Clases reales.
        pred: Clases predichas.

    Returns:
        exactitud (6 clases), exactitud_grupo y recall_grupos (promedio del recall
        por grupo: la métrica de decisión, insensible al desbalance).
    """
    if not len(y):
        return {"n": 0, "exactitud": float("nan"), "exactitud_grupo": float("nan"),
                "recall_grupos": float("nan")}
    g_real = np.array([GRUPO_CLASE[CLASES[k]] for k in y])
    g_pred = np.array([GRUPO_CLASE[CLASES[k]] for k in pred])
    recalls = [float((g_pred[g_real == g] == g).mean()) for g in GRUPOS if (g_real == g).any()]
    return {"n": int(len(y)), "exactitud": float((y == pred).mean()),
            "exactitud_grupo": float((g_real == g_pred).mean()),
            "recall_grupos": float(np.mean(recalls))}


def evaluar(datos: dict[str, np.ndarray], actual: Cabeza, nueva: Cabeza) -> dict:
    """
    Compara el modelo actual y el nuevo en el conjunto de PRUEBA.

    Args:
        datos: Características y metadatos.
        actual: Capa final del modelo actual.
        nueva: Capa final entrenada.

    Returns:
        Diccionario {fuente: {condicion: {"actual": métricas, "nuevo": métricas}}} y,
        bajo la clave "criticas", el recall por categoría crítica.
    """
    pru = datos["conjunto"] == "prueba"
    X, y = datos["X"][pru], datos["y"][pru]
    pred = {"actual": actual.logits(X).argmax(1), "nuevo": nueva.logits(X).argmax(1)}
    fuente, condicion = datos["fuente"][pru], datos["condicion"][pru]
    resultado: dict = {}
    for f in sorted(set(fuente.tolist())):
        for c in ("limpia", "webcam"):
            m = (fuente == f) & (condicion == c)
            resultado.setdefault(f, {})[c] = {k: metricas(y[m], p[m]) for k, p in pred.items()}

    criticas: dict[str, dict] = {}
    detalle, critica = datos["detalle"][pru], datos["critica"][pru]
    g_real = np.array([GRUPO_CLASE[CLASES[k]] for k in y])
    for cat in sorted(set(detalle[critica].tolist())):
        m = critica & (detalle == cat)
        criticas[cat] = {"n": int(m.sum())} | {
            k: float((np.array([GRUPO_CLASE[CLASES[j]] for j in p[m]]) == g_real[m]).mean())
            for k, p in pred.items()}
    resultado["criticas"] = criticas
    return resultado


def decidir(res: dict, tolerancia: float) -> tuple[bool, list[str]]:
    """
    Regla de aceptación del modelo nuevo.

    Acepta si el recall por grupos en las fuentes NUEVAS (TACO / fotos propias,
    promedio de condiciones limpia y webcam) sube, y la exactitud en TrashNet no cae
    más que `tolerancia`.

    Args:
        res: Salida de `evaluar`.
        tolerancia: Caída máxima permitida en TrashNet (fracción, 0.03 = 3 puntos).

    Returns:
        (aceptado, motivos legibles).
    """
    motivos, ok = [], True
    nuevas = [f for f in res if f in ("taco", "fotos")]
    for f in nuevas:
        a = np.nanmean([res[f][c]["actual"]["recall_grupos"] for c in ("limpia", "webcam")])
        n = np.nanmean([res[f][c]["nuevo"]["recall_grupos"] for c in ("limpia", "webcam")])
        mejora = n > a
        ok &= mejora
        motivos.append(f"{'✅' if mejora else '❌'} {f}: recall por grupos {a:.1%} -> {n:.1%}")
    if "trashnet" in res:
        a = res["trashnet"]["limpia"]["actual"]["exactitud"]
        n = res["trashnet"]["limpia"]["nuevo"]["exactitud"]
        estable = n >= a - tolerancia
        ok &= estable
        motivos.append(f"{'✅' if estable else '❌'} trashnet: exactitud {a:.1%} -> {n:.1%} "
                       f"(caída máxima permitida {tolerancia:.0%})")
    if not nuevas:
        ok = False
        motivos.append("❌ No hay fotos de prueba de TACO ni propias: no se puede demostrar mejora.")
    return bool(ok), motivos


def formatear_reporte(res: dict, C: float, acc_val: float, motivos: list[str],
                      aceptado: bool, datos: dict[str, np.ndarray]) -> str:
    """Construye el reporte de texto (consola y models/reporte_reentrenamiento.txt)."""
    lineas = [f"Reentrenamiento — {datetime.now():%Y-%m-%d %H:%M}",
              f"C elegido = {C}  |  exactitud balanceada en validación = {acc_val:.3f}", ""]
    conteo = Counter(zip(datos["fuente"].tolist(), datos["conjunto"].tolist()))
    lineas.append("Imágenes por fuente y conjunto (incluye variantes aumentadas):")
    for f in sorted({k[0] for k in conteo}):
        lineas.append(f"   {f:<9} " + "  ".join(f"{c}={conteo[(f, c)]:,}" for c in CONJUNTOS))
    lineas += ["", "Comparación en PRUEBA (fotos nunca vistas)",
               f"   {'fuente':<9}{'condición':<10}{'n':>6}   {'exactitud':>21}   "
               f"{'recall por grupos':>21}",
               f"   {'':<25}   {'actual':>9} -> {'nuevo':<9}   {'actual':>9} -> {'nuevo':<9}"]
    for f, por_cond in res.items():
        if f == "criticas":
            continue
        for c, m in por_cond.items():
            a, n = m["actual"], m["nuevo"]
            lineas.append(f"   {f:<9}{c:<10}{a['n']:>6}   {a['exactitud']:>9.1%} -> {n['exactitud']:<9.1%}"
                          f"   {a['recall_grupos']:>9.1%} -> {n['recall_grupos']:<9.1%}")
    if res.get("criticas"):
        lineas += ["", "Categorías críticas (acierto de grupo en prueba)"]
        for cat, m in sorted(res["criticas"].items(), key=lambda kv: -kv[1]["n"]):
            lineas.append(f"   {cat[:30]:<30} n={m['n']:<5} {m['actual']:6.1%} -> {m['nuevo']:6.1%}")
    lineas += ["", "Decisión:", *[f"   {m}" for m in motivos],
               f"   => {'MODELO NUEVO ACEPTADO' if aceptado else 'SE CONSERVA EL MODELO ACTUAL'}"]
    return "\n".join(lineas)


def instalar_modelo(candidato: Path, metricas_json: dict, reporte: str) -> Path:
    """
    Respalda el modelo actual y pone el candidato en su lugar.

    Args:
        candidato: Modelo nuevo.
        metricas_json: Metadatos a escribir junto al modelo.
        reporte: Texto del reporte.

    Returns:
        Carpeta del respaldo creado.
    """
    respaldo = DIR_MODELOS / "respaldo" / f"{datetime.now():%Y%m%d_%H%M%S}"
    respaldo.mkdir(parents=True, exist_ok=True)
    for archivo in (MODELO_APP, MODELO_APP.with_suffix(".json"),
                    DIR_MODELOS / "reporte_reentrenamiento.txt"):
        if archivo.is_file():
            shutil.copy2(archivo, respaldo / archivo.name)
    shutil.copy2(candidato, MODELO_APP)
    MODELO_APP.with_suffix(".json").write_text(json.dumps(metricas_json, indent=2,
                                                          ensure_ascii=False), encoding="utf-8")
    (DIR_MODELOS / "reporte_reentrenamiento.txt").write_text(reporte, encoding="utf-8")
    return respaldo


def restaurar(carpeta: Path) -> int:
    """
    Devuelve a la app un modelo respaldado.

    Args:
        carpeta: Carpeta dentro de models/respaldo/.

    Returns:
        Código de salida del proceso.
    """
    modelo = carpeta / MODELO_APP.name
    if not modelo.is_file():
        sys.exit(f"❌ {carpeta} no contiene {MODELO_APP.name}")
    for archivo in carpeta.iterdir():
        shutil.copy2(archivo, DIR_MODELOS / archivo.name)
    print(f"↩️  Modelo restaurado desde {carpeta}")
    return 0


# ===========================================================================
# 6. PROGRAMA PRINCIPAL
# ===========================================================================
def ejecutar(args: argparse.Namespace) -> int:
    """
    Ejecuta el flujo completo: red base -> datos -> características -> entrenamiento
    -> comparación -> decisión.

    Args:
        args: Argumentos de línea de comandos.

    Returns:
        Código de salida (0 = terminó; el modelo puede o no haberse reemplazado).
    """
    if not MODELO_APP.is_file():
        sys.exit(f"❌ No se encontró el modelo actual: {MODELO_APP}")
    sello = f"{datetime.now():%Y%m%d_%H%M%S}"

    print("1️⃣  Extrayendo la red base del modelo actual...")
    ruta_backbone, cabeza_actual = extraer_backbone(MODELO_APP, DIR_CACHE / "backbone.onnx")
    extractor = Extractor(ruta_backbone, args.tam_lote)

    print("2️⃣  Reuniendo datos...")
    ejemplos: list[Ejemplo] = []
    if args.trashnet:
        ejemplos += cargar_trashnet(Path(args.trashnet).resolve())
    if args.taco:
        ejemplos += cargar_coco(Path(args.taco).resolve(), normalizar_lotes(args.lotes_validacion),
                                normalizar_lotes(args.lotes_prueba), args)
    if args.fotos:
        ejemplos += cargar_fotos(Path(args.fotos).resolve())
    if args.limite:
        rng = np.random.default_rng(args.semilla)
        ejemplos = [ejemplos[i] for i in sorted(rng.permutation(len(ejemplos))[:args.limite])]
    resumen = Counter((e.fuente, e.conjunto) for e in ejemplos)
    for f in sorted({e.fuente for e in ejemplos}):
        print(f"   {f:<9} " + "  ".join(f"{c}={resumen[(f, c)]:,}" for c in CONJUNTOS))

    print(f"3️⃣  Extrayendo características (original + {args.aumentos} aumentos por foto de "
          "entrenamiento)...")
    datos = construir_caracteristicas(ejemplos, extractor, _huella(ejemplos, ruta_backbone, args), args)

    print("4️⃣  Entrenando...")
    cabeza_nueva, C, acc_val = entrenar(datos, args.grilla_c)

    candidato = exportar_modelo(MODELO_APP, cabeza_nueva, DIR_SALIDAS / f"modelo_{sello}.onnx")
    prueba_eq = [ip.recortar(ip.leer_imagen(e.muestra.ruta, e.muestra.tam_anotado),
                             e.muestra.bbox, args.margen) for e in ejemplos[:3]]
    dif = max(verificar_equivalencia(MODELO_APP, cabeza_actual, extractor, prueba_eq),
              verificar_equivalencia(candidato, cabeza_nueva, extractor, prueba_eq))
    if dif > 1e-2:
        sys.exit(f"❌ Verificación fallida: el ONNX exportado difiere {dif:.4f} del cálculo esperado.")

    print("5️⃣  Comparando modelo actual vs. nuevo en fotos de prueba...")
    res = evaluar(datos, cabeza_actual, cabeza_nueva)
    aceptado, motivos = decidir(res, args.tolerancia)
    aceptado = (aceptado or args.forzar) and not args.no_reemplazar
    reporte = formatear_reporte(res, C, acc_val, motivos, aceptado, datos)
    print("\n" + "=" * 78 + "\n" + reporte + "\n" + "=" * 78)
    (DIR_SALIDAS / f"reporte_{sello}.txt").write_text(reporte, encoding="utf-8")

    metricas_json = {
        "reentrenado": sello, "C": C, "exactitud_balanceada_validacion": round(acc_val, 4),
        "clases": CLASES, "fuentes": sorted({e.fuente for e in ejemplos}),
        "lotes_prueba": sorted(normalizar_lotes(args.lotes_prueba)) if args.taco else [],
        "prueba": {f: {c: m["nuevo"] for c, m in v.items()} for f, v in res.items() if f != "criticas"},
    }
    if aceptado:
        respaldo = instalar_modelo(candidato, metricas_json, reporte)
        print(f"\n✅ Modelo nuevo instalado en {MODELO_APP.relative_to(RAIZ)}."
              f"\n   Respaldo del anterior: {respaldo.relative_to(RAIZ)}"
              f"\n   Para deshacer: python entrenamiento/reentrenar.py --restaurar "
              f"{respaldo.relative_to(RAIZ)}")
    else:
        print(f"\nℹ️  La app sigue con el modelo actual. Candidato guardado en "
              f"{candidato.relative_to(RAIZ)}" + ("" if args.no_reemplazar else
                                                  " (use --forzar para instalarlo de todos modos)."))
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Punto de entrada de la línea de comandos.

    Args:
        argv: Argumentos (None = `sys.argv[1:]`).

    Returns:
        Código de salida del proceso.
    """
    p = argparse.ArgumentParser(description="Reentrena el clasificador con TACO + TrashNet + fotos propias.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    defecto_trashnet = RAIZ / "entrenamiento" / "dataset-resized"
    p.add_argument("--taco", help="Carpeta COCO (la que contiene annotations.json).")
    p.add_argument("--trashnet", default=str(defecto_trashnet) if defecto_trashnet.is_dir() else None,
                   help="Carpeta dataset-resized de TrashNet (evita que el modelo olvide lo aprendido).")
    p.add_argument("--fotos", help="Fotos propias en subcarpetas por clase (icopor/, metal/, ...).")
    p.add_argument("--sin-trashnet", action="store_true",
                   help="Permite entrenar sin TrashNet (no recomendado).")
    p.add_argument("--lotes-validacion", default="12,13", help="Lotes TACO para elegir C.")
    p.add_argument("--lotes-prueba", default="14,15", help="Lotes TACO para la comparación final.")
    p.add_argument("--aumentos", type=int, default=2, help="Variantes aumentadas por foto de entrenamiento.")
    p.add_argument("--margen", type=float, default=0.30, help="Contexto alrededor de cada caja.")
    p.add_argument("--lado-minimo", type=int, default=48, help="Ignora objetos más pequeños (px).")
    p.add_argument("--max-por-imagen", type=int, default=3, help="Objetos por foto de TACO.")
    p.add_argument("--grilla-c", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0])
    p.add_argument("--tolerancia", type=float, default=0.03,
                   help="Caída máxima aceptada en TrashNet (0.03 = 3 puntos).")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--limite", type=int, default=0, help="Usa solo N ejemplos (prueba rápida).")
    p.add_argument("--tam-lote", type=int, default=32, help="Imágenes por inferencia.")
    p.add_argument("--no-reemplazar", action="store_true", help="Solo entrena y compara.")
    p.add_argument("--forzar", action="store_true", help="Instala el modelo nuevo aunque no mejore.")
    p.add_argument("--sin-cache", action="store_true", help="Recalcula las características.")
    p.add_argument("--restaurar", help="Restaura un respaldo de models/respaldo/<fecha> y termina.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.restaurar:
        return restaurar(Path(args.restaurar).resolve())
    if not (args.taco or args.fotos):
        p.error("Indique --taco y/o --fotos: son los datos nuevos que corrigen la brecha de dominio.")
    if not args.trashnet and not args.sin_trashnet:
        p.error("No se encontró TrashNet. Descargue dataset-resized.zip de "
                "https://github.com/garythung/trashnet, descomprímalo en entrenamiento/dataset-resized "
                "o indique --trashnet <ruta>. (Use --sin-trashnet bajo su responsabilidad.)")
    if args.aumentos < 0:
        p.error("--aumentos no puede ser negativo.")
    return ejecutar(args)


if __name__ == "__main__":
    sys.exit(main())
