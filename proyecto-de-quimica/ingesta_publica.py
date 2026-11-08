"""
ingesta_publica.py — Ingesta COCO (TACO / Roboflow), aumento de datos y registro masivo
========================================================================================
Cierra la «brecha de dominio» del motor de visión frente al consumo colombiano
(empaques multicapa metalizados, icopor, aerosoles y envases de aseo) en tres pasos:

    1. INGESTA     Lee un directorio local con imágenes y anotaciones MS COCO
                   (TACO, exportaciones COCO de Roboflow, o cualquier dataset COCO)
                   usando la API oficial `pycocotools`. Cada bbox se recorta como si
                   el objeto estuviera frente a la cámara.
    2. MAPEO       Traduce categorías y súper-categorías del dataset a las clases
                   normativas del sistema (Res. 2184 de 2019), p. ej.
                   crisp_packet -> «No aprovechable (Multicapa)».
    3. AUMENTO     Con `albumentations` genera variantes que reproducen la webcam:
                   encuadre imperfecto (RandomCrop), destellos especulares sobre
                   empaques metalizados (RandomSunFlare), exposición automática
                   (RandomBrightnessContrast), ruido de sensor, movimiento y
                   compresión VP8/JPEG del flujo WebRTC.
    4. EVALUACIÓN  Cada variante pasa por el MISMO `ClasificadorResiduos` que usa la
                   cámara y el resultado se guarda en SQLite (modo WAL, inserción por
                   lotes e idempotente) para poblar el Tablero Estadístico.

Formatos de entrada (detección automática con --ruta):
    * TACO:      <ruta>/annotations.json + <ruta>/batch_*/xxxx.jpg
                 https://github.com/pedropro/TACO  (CC BY 4.0) -> python download.py
    * Roboflow:  <ruta>/{train,valid,test}/_annotations.coco.json + imágenes al lado
    * COCO:      cualquier *.json con claves images/annotations/categories (--anotaciones)
    * TrashNet:  <ruta>/{cardboard,glass,metal,paper,plastic,trash}/*.jpg
    * Carpeta:   <ruta>/<clase>/*.jpg; el nombre de la subcarpeta se mapea con el mismo
                 diccionario (sirve para fotos propias: icopor/, paquete_papas/, ...)

Origen en la BD:
    Las imágenes «originales» se registran con origen=<dataset> y las aumentadas con
    origen=<dataset>_aug. Así el tablero compara la exactitud en condiciones de
    laboratorio frente a condiciones de webcam (robustez del modelo).

Ejemplos:
    python ingesta_publica.py --ruta D:/datasets/TACO/data --aumentos 2 --distribuir-dias 30
    python ingesta_publica.py --ruta D:/datasets/TACO/data --solo-criticas --dry-run
    python ingesta_publica.py --ruta D:/roboflow/snacks-co --origen roboflow --guardar-todo
    python ingesta_publica.py --ruta D:/datasets/TACO/data --exportar-recortes D:/finetune
    python ingesta_publica.py --borrar-origen taco_aug

⚠️ Fuga de datos: la MobileNetV2 del proyecto se entrenó con TrashNet; para TrashNet use
--solo-prueba. Si exporta recortes para reentrenar (--exportar-recortes), no valide luego
con las mismas fotos: separe por imagen (p. ej. lotes TACO distintos o splits Roboflow).

Autor: Johan Lasso
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import random
import re
import sys
import time
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image, ImageOps

import database as db
from vision_engine import ClasificadorResiduos

try:  # Dependencias de la ingesta: se informan con un mensaje claro si faltan.
    import albumentations as A
    from pycocotools.coco import COCO
except ImportError as _exc:  # pragma: no cover - depende del entorno
    sys.exit(f"❌ Falta una dependencia de la ingesta ({_exc.name}). "
             "Ejecute:  pip install -r requirements.txt")

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MOTORES = ["Motor Híbrido Estricto", "Motor IA (MobileNetV2)",
           "Motor Geométrico (Forma/Contornos)", "Motor Básico (Color)"]
SUFIJO_AUMENTADO = "_aug"


# ===========================================================================
# 1. CLASES NORMATIVAS (Res. 2184 de 2019) Y MAPEO DE CATEGORÍAS
# ===========================================================================
@dataclass(frozen=True)
class ClaseNormativa:
    """
    Clase de destino del sistema de reciclaje.

    Attributes:
        nombre: Etiqueta que se guarda en `etiqueta_real` (ej. «No aprovechable (Icopor)»).
        contenedor: Contenedor correcto según la Res. 2184 (Blanco / Negro / Verde) o
            None si el residuo no va a ningún contenedor doméstico (posconsumo).
        evaluable: True si el motor puede predecir esa clase. Las no evaluables
            (orgánico, peligroso, sin etiqueta) se clasifican y registran, pero no
            cuentan para la exactitud (se guardan con etiqueta_real = NULL).
        metalizada: Superficie reflectiva: aumenta la probabilidad de destellos.
        clase_trashnet: Carpeta destino al exportar recortes para reentrenar la red
            (mismas 6 clases que `entrenamiento/entrenar_trashnet.py`).
    """

    nombre: str
    contenedor: str | None
    evaluable: bool = True
    metalizada: bool = False
    clase_trashnet: str | None = None

    @property
    def grupo(self) -> str:
        """Grupo comparable con la salida del motor (ver `database.grupo_material`)."""
        return db.grupo_material(self.nombre)


def _clase(nombre: str, contenedor: str | None, trashnet: str | None, *,
           evaluable: bool = True, metalizada: bool = False) -> ClaseNormativa:
    """Atajo de construcción para el catálogo `CLASES_NORMATIVAS`."""
    return ClaseNormativa(nombre, contenedor, evaluable, metalizada, trashnet)


CLASES_NORMATIVAS: dict[str, ClaseNormativa] = {c.nombre: c for c in (
    _clase("Plástico (Blanco)", "Blanco", "plastic"),
    _clase("Metal (Blanco)", "Blanco", "metal", metalizada=True),
    _clase("Vidrio (Blanco)", "Blanco", "glass"),
    _clase("Papel (Blanco)", "Blanco", "paper"),
    _clase("Cartón (Blanco)", "Blanco", "cardboard"),
    # Tetra Pak: multicapa RÍGIDO, sí aprovechable en Colombia (ruta de recolección de cartón).
    _clase("Cartón (Tetra Pak)", "Blanco", "cardboard"),
    # Multicapa FLEXIBLE metalizado (papas, galletas, snacks): PET/BOPP + aluminio + PE.
    _clase("No aprovechable (Multicapa)", "Negro", "trash", metalizada=True),
    # Poliestireno expandido: técnicamente reciclable, en la práctica llega contaminado.
    _clase("No aprovechable (Icopor)", "Negro", "trash"),
    _clase("No aprovechable (Sanitario)", "Negro", "trash"),
    _clase("No aprovechable (Plástico de un solo uso)", "Negro", "trash"),
    _clase("No aprovechable", "Negro", "trash"),
    _clase("Orgánico (Verde)", "Verde", None, evaluable=False),
    _clase("Peligroso (Posconsumo)", None, None, evaluable=False),
    _clase("Sin etiqueta", None, None, evaluable=False),
)}


def normalizar_nombre(texto: str) -> str:
    """
    Normaliza un nombre de categoría a snake_case ASCII.

    Permite usar una sola clave para TACO («Crisp packet»), Roboflow
    («crisp-packet», «Crisp_Packet») y nombres en español («Envase plástico»).

    >>> normalizar_nombre("Foam food container")
    'foam_food_container'
    >>> normalizar_nombre("Cartón  Ondulado")
    'carton_ondulado'
    """
    ascii_ = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_.lower()).strip("_")


# --- Mapeo específico por CATEGORÍA (prioridad 1) --------------------------
# Las 60 categorías oficiales de TACO + alias frecuentes de Roboflow y del
# ecosistema de consumo colombiano. Claves ya normalizadas con `normalizar_nombre`.
_MULTICAPA = "No aprovechable (Multicapa)"
_ICOPOR = "No aprovechable (Icopor)"
_METAL = "Metal (Blanco)"
_PLASTICO = "Plástico (Blanco)"
_UN_USO = "No aprovechable (Plástico de un solo uso)"

MAPEO_CATEGORIAS: dict[str, str] = {
    # ---- Clases críticas: empaques flexibles multicapa -------------------
    "crisp_packet": _MULTICAPA,
    "chips_bag": _MULTICAPA, "chip_bag": _MULTICAPA, "snack_bag": _MULTICAPA,
    "snack_wrapper": _MULTICAPA, "snack_packet": _MULTICAPA, "candy_wrapper": _MULTICAPA,
    "cookie_wrapper": _MULTICAPA, "biscuit_wrapper": _MULTICAPA, "metallized_film": _MULTICAPA,
    "multilayer_packaging": _MULTICAPA, "flexible_packaging": _MULTICAPA,
    "paquete_papas": _MULTICAPA, "paquete_de_papas": _MULTICAPA, "empaque_snack": _MULTICAPA,
    "empaque_galletas": _MULTICAPA, "envoltura_galletas": _MULTICAPA, "multicapa": _MULTICAPA,
    "aluminium_blister_pack": _MULTICAPA, "carded_blister_pack": _MULTICAPA,
    "squeezable_tube": _MULTICAPA,            # Crema dental: laminado plástico-aluminio
    "plastified_paper_bag": _MULTICAPA,
    # ---- Clases críticas: poliestireno expandido (icopor) ---------------
    "foam_food_container": _ICOPOR, "foam_cup": _ICOPOR, "styrofoam_piece": _ICOPOR,
    "styrofoam": _ICOPOR, "polystyrene": _ICOPOR, "expanded_polystyrene": _ICOPOR,
    "eps": _ICOPOR, "foam": _ICOPOR, "foam_tray": _ICOPOR, "clamshell": _ICOPOR,
    "icopor": _ICOPOR, "poliestireno": _ICOPOR, "bandeja_icopor": _ICOPOR,
    "coca_icopor": _ICOPOR, "vaso_icopor": _ICOPOR,
    # ---- Clases críticas: aerosoles (acero/aluminio, deben ir VACÍOS) ----
    "aerosol": _METAL, "aerosol_can": _METAL, "spray_can": _METAL, "spray": _METAL,
    "desodorante_aerosol": _METAL, "insecticida": _METAL, "ambientador": _METAL,
    # ---- Envases de productos de aseo (PEAD/PP, marcas de tiendas de descuento)
    "detergent_bottle": _PLASTICO, "cleaning_product": _PLASTICO, "bleach_bottle": _PLASTICO,
    "shampoo_bottle": _PLASTICO, "envase_aseo": _PLASTICO, "galon": _PLASTICO,
    "cloro": _PLASTICO, "blanqueador": _PLASTICO, "suavizante": _PLASTICO,
    "jabon_liquido": _PLASTICO, "detergente": _PLASTICO, "limpiapisos": _PLASTICO,
    # ---- Resto de TACO ---------------------------------------------------
    "aluminium_foil": _METAL, "metal_bottle_cap": _METAL, "food_can": _METAL,
    "drink_can": _METAL, "metal_lid": _METAL, "pop_tab": _METAL, "scrap_metal": _METAL,
    "other_plastic_bottle": _PLASTICO, "clear_plastic_bottle": _PLASTICO,
    "plastic_bottle_cap": _PLASTICO, "disposable_plastic_cup": _PLASTICO,
    "other_plastic_cup": _PLASTICO, "plastic_lid": _PLASTICO, "other_plastic": _PLASTICO,
    "spread_tub": _PLASTICO, "tupperware": _PLASTICO, "disposable_food_container": _PLASTICO,
    "other_plastic_container": _PLASTICO,
    "glass_bottle": "Vidrio (Blanco)", "broken_glass": "Vidrio (Blanco)",
    "glass_cup": "Vidrio (Blanco)", "glass_jar": "Vidrio (Blanco)",
    "toilet_tube": "Cartón (Blanco)", "other_carton": "Cartón (Blanco)",
    "egg_carton": "Cartón (Blanco)", "corrugated_carton": "Cartón (Blanco)",
    "meal_carton": "Cartón (Blanco)", "pizza_box": "Cartón (Blanco)",
    "drink_carton": "Cartón (Tetra Pak)", "tetra_pak": "Cartón (Tetra Pak)",
    "magazine_paper": "Papel (Blanco)", "wrapping_paper": "Papel (Blanco)",
    "normal_paper": "Papel (Blanco)", "paper_bag": "Papel (Blanco)",
    "paper_cup": "No aprovechable",           # Papel con película de PE
    "paper_straw": "No aprovechable",
    "tissues": "No aprovechable (Sanitario)", "cigarette": "No aprovechable (Sanitario)",
    "plastic_glooves": "No aprovechable (Sanitario)", "shoe": "No aprovechable",
    "rope_strings": "No aprovechable",
    "plastic_film": _UN_USO, "six_pack_rings": _UN_USO, "garbage_bag": _UN_USO,
    "other_plastic_wrapper": _UN_USO, "single_use_carrier_bag": _UN_USO,
    "polypropylene_bag": _UN_USO, "plastic_utensils": _UN_USO, "plastic_straw": _UN_USO,
    "food_waste": "Orgánico (Verde)", "battery": "Peligroso (Posconsumo)",
    "unlabeled_litter": "Sin etiqueta",
    # ---- Carpetas TrashNet / genéricas -----------------------------------
    "cardboard": "Cartón (Blanco)", "paper": "Papel (Blanco)", "glass": "Vidrio (Blanco)",
    "metal": _METAL, "plastic": _PLASTICO, "trash": "No aprovechable",
}

# --- Mapeo de respaldo por SÚPER-CATEGORÍA (prioridad 2) -------------------
# Se usa cuando la categoría no aparece arriba (datasets Roboflow con nombres propios).
MAPEO_SUPERCATEGORIAS: dict[str, str] = {
    "plastic_bag_wrapper": _UN_USO, "styrofoam": _ICOPOR, "foam": _ICOPOR,
    "can": _METAL, "aluminium_foil": _METAL, "bottle_cap": _PLASTICO, "lid": _PLASTICO,
    "scrap_metal": _METAL, "pop_tab": _METAL, "metal": _METAL,
    "bottle": _PLASTICO, "plastic_container": _PLASTICO, "other_plastic": _PLASTICO,
    "plastic": _PLASTICO, "cup": _UN_USO, "straw": _UN_USO, "plastic_utensils": _UN_USO,
    "broken_glass": "Vidrio (Blanco)", "glass_jar": "Vidrio (Blanco)", "glass": "Vidrio (Blanco)",
    "carton": "Cartón (Blanco)", "cardboard": "Cartón (Blanco)",
    "paper": "Papel (Blanco)", "paper_bag": "Papel (Blanco)",
    "blister_pack": _MULTICAPA, "squeezable_tube": _MULTICAPA, "multilayer": _MULTICAPA,
    "food_waste": "Orgánico (Verde)", "organic": "Orgánico (Verde)",
    "battery": "Peligroso (Posconsumo)", "hazardous": "Peligroso (Posconsumo)",
    "cigarette": "No aprovechable (Sanitario)", "plastic_glooves": "No aprovechable (Sanitario)",
    "shoe": "No aprovechable", "rope_strings": "No aprovechable", "trash": "No aprovechable",
    "unlabeled_litter": "Sin etiqueta",
}

# Categorías que materializan la brecha de dominio (reporte de recall dedicado).
CATEGORIAS_CRITICAS: frozenset[str] = frozenset(
    k for k, v in MAPEO_CATEGORIAS.items() if v in (_MULTICAPA, _ICOPOR)
) | {"aerosol", "aerosol_can", "spray_can", "spray", "desodorante_aerosol", "insecticida",
     "ambientador", "detergent_bottle", "cleaning_product", "bleach_bottle", "envase_aseo",
     "galon", "cloro", "blanqueador", "suavizante", "jabon_liquido", "detergente", "limpiapisos"}


def _validar_mapeos() -> None:
    """Garantiza en el arranque que todo destino del mapeo es una clase normativa."""
    for nombre, mapeo in (("MAPEO_CATEGORIAS", MAPEO_CATEGORIAS),
                          ("MAPEO_SUPERCATEGORIAS", MAPEO_SUPERCATEGORIAS)):
        malos = {v for v in mapeo.values() if v not in CLASES_NORMATIVAS}
        if malos:
            raise RuntimeError(f"{nombre} apunta a clases inexistentes: {sorted(malos)}")


_validar_mapeos()


def mapear_categoria(nombre: str, supercategoria: str | None = None) -> ClaseNormativa | None:
    """
    Resuelve la clase normativa de una categoría del dataset.

    Orden de resolución: categoría exacta -> súper-categoría -> None (sin mapeo).

    Args:
        nombre: Nombre de la categoría (`categories[].name` en COCO).
        supercategoria: `categories[].supercategory` (opcional).

    Returns:
        La `ClaseNormativa` correspondiente o None si no hay equivalencia.
    """
    destino = MAPEO_CATEGORIAS.get(normalizar_nombre(nombre))
    if destino is None and supercategoria:
        destino = MAPEO_SUPERCATEGORIAS.get(normalizar_nombre(supercategoria))
    return CLASES_NORMATIVAS.get(destino) if destino else None


# ===========================================================================
# 2. MUESTRAS Y LECTURA DE IMÁGENES
# ===========================================================================
@dataclass(frozen=True)
class Muestra:
    """
    Un objeto a evaluar: una anotación COCO o una imagen completa.

    Attributes:
        archivo: Identificador estable (ruta relativa [+ «#id_anotación»]). Junto con
            origen y motor forma la clave única de idempotencia en SQLite.
        ruta: Ruta absoluta de la imagen en disco.
        bbox: Caja COCO (x, y, ancho, alto) en píxeles o None para la imagen completa.
        tam_anotado: (ancho, alto) declarado en el JSON, para resolver la orientación EXIF.
        clase: Clase normativa (verdad de terreno) o None si no tiene mapeo.
        detalle: Categoría original del dataset (para reportes y CSV).
        critica: True si pertenece a las clases de la brecha de dominio.
    """

    archivo: str
    ruta: Path
    bbox: tuple[float, float, float, float] | None
    tam_anotado: tuple[int, int] | None
    clase: ClaseNormativa | None
    detalle: str
    critica: bool = False


@lru_cache(maxsize=8)
def leer_imagen(ruta: Path, tam_anotado: tuple[int, int] | None = None) -> np.ndarray:
    """
    Lee una imagen como BGR respetando la orientación con la que se anotó.

    Se usa PIL en lugar de `cv2.imread` porque soporta rutas con tildes en Windows.
    Las fotos de celular (TACO) traen la etiqueta EXIF de orientación; casi siempre
    las cajas están sobre la imagen ya rotada, pero algunas herramientas anotan sobre
    los píxeles crudos. Si el JSON declara su tamaño, se elige la versión cuyo
    (ancho, alto) coincide. La caché evita releer la foto para cada objeto.

    Args:
        ruta: Ruta de la imagen.
        tam_anotado: (ancho, alto) según el JSON COCO, si se conoce.

    Returns:
        Arreglo BGR uint8 de solo lectura (no modificar: es compartido por la caché).
    """
    with Image.open(ruta) as img:
        orientada = ImageOps.exif_transpose(img)
        if tam_anotado and orientada.size != tam_anotado and img.size == tam_anotado:
            orientada = img
        bgr = cv2.cvtColor(np.asarray(orientada.convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr.setflags(write=False)
    return bgr


def recortar(bgr: np.ndarray, bbox: tuple[float, float, float, float] | None,
             margen: float) -> np.ndarray:
    """
    Recorta el objeto con un margen de contexto alrededor de la caja.

    El margen deja fondo suficiente para que `RandomCrop` simule un encuadre
    imperfecto sin cortar el objeto (con margen 0.30 el objeto ocupa el ~62 %
    central y el recorte aleatorio retira como máximo el 12 % del lado).

    Args:
        bgr: Imagen completa.
        bbox: Caja COCO (x, y, w, h) o None para devolver la imagen completa.
        margen: Fracción del ancho/alto de la caja que se añade por cada lado.

    Returns:
        Recorte BGR (copia contigua, apto para escribir).

    Raises:
        ValueError: Si la caja cae fuera de la imagen.
    """
    if bbox is None:
        return np.ascontiguousarray(bgr)
    alto, ancho = bgr.shape[:2]
    x, y, w, h = bbox
    px, py = w * margen, h * margen
    x1, y1 = max(0, int(x - px)), max(0, int(y - py))
    x2, y2 = min(ancho, int(np.ceil(x + w + px))), min(alto, int(np.ceil(y + h + py)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError(f"bbox fuera de la imagen o degenerada: {bbox}")
    return np.ascontiguousarray(bgr[y1:y2, x1:x2])


def buscar_anotaciones(ruta: Path) -> list[Path]:
    """
    Localiza los JSON COCO dentro de `ruta` (TACO o splits de Roboflow).

    Args:
        ruta: Carpeta raíz del dataset.

    Returns:
        Lista ordenada de archivos de anotaciones (vacía si no hay).
    """
    if (ruta / "data" / "annotations.json").is_file() and not any(ruta.glob("batch_*")):
        return [ruta / "data" / "annotations.json"]
    directos = [ruta / "annotations.json"] if (ruta / "annotations.json").is_file() else []
    return directos or sorted(ruta.rglob("*_annotations.coco.json")) or sorted(
        p for p in ruta.glob("*.json") if _parece_coco(p))


def _parece_coco(ruta: Path) -> bool:
    """Heurística barata: el inicio del JSON contiene la clave «images»."""
    try:
        with open(ruta, encoding="utf-8") as f:
            return '"images"' in f.read(4096)
    except OSError:
        return False


def cargar_coco(ruta_json: Path) -> COCO:
    """
    Carga un archivo COCO con `pycocotools` silenciando sus mensajes de consola.

    Args:
        ruta_json: Archivo de anotaciones.

    Returns:
        Objeto `COCO` con los índices imgs/anns/cats ya construidos.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(ruta_json))


def muestras_coco(rutas_json: list[Path], raiz: Path, *, lado_minimo: int,
                  max_por_imagen: int, incluir_sin_mapeo: bool) -> tuple[list[Muestra], Counter]:
    """
    Convierte uno o varios archivos COCO en muestras recortables.

    Las rutas de imagen (`file_name`) se resuelven relativas a la carpeta del JSON,
    convención compartida por TACO y Roboflow. Se descartan anotaciones `iscrowd`,
    cajas menores que `lado_minimo` y, por imagen, se conservan los objetos más
    grandes (los que una persona acercaría a la cámara).

    Args:
        rutas_json: Archivos de anotaciones.
        raiz: Carpeta del dataset (base de los identificadores `archivo`).
        lado_minimo: Lado mínimo de la caja en píxeles.
        max_por_imagen: Máximo de objetos por fotografía.
        incluir_sin_mapeo: Si True, procesa también categorías sin clase normativa.

    Returns:
        (muestras, conteo de categorías sin mapeo).
    """
    muestras: list[Muestra] = []
    sin_mapeo: Counter = Counter()
    faltantes = 0
    for ruta_json in rutas_json:
        coco = cargar_coco(ruta_json)
        base = ruta_json.parent
        clases = {cid: (cat["name"], mapear_categoria(cat["name"], cat.get("supercategory")))
                  for cid, cat in coco.cats.items()}
        for id_img in coco.getImgIds():
            meta = coco.imgs[id_img]
            ruta_img = base / meta["file_name"]
            if not ruta_img.is_file():
                if (base / "data" / meta["file_name"]).is_file():
                    ruta_img = base / "data" / meta["file_name"]
                else:
                    faltantes += 1
                    continue
            anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=id_img, iscrowd=False))
                    if len(a.get("bbox") or ()) == 4 and min(a["bbox"][2:]) >= lado_minimo]
            anns.sort(key=lambda a: a.get("area") or a["bbox"][2] * a["bbox"][3], reverse=True)
            tam = (int(meta["width"]), int(meta["height"])) if meta.get("width") else None
            for an in anns[:max_por_imagen]:
                nombre, clase = clases.get(an["category_id"], (str(an["category_id"]), None))
                if clase is None:
                    sin_mapeo[nombre] += 1
                    if not incluir_sin_mapeo:
                        continue
                muestras.append(Muestra(
                    archivo=f"{ruta_img.relative_to(raiz).as_posix()}#{an['id']}",
                    ruta=ruta_img, bbox=tuple(float(v) for v in an["bbox"]), tam_anotado=tam,
                    clase=clase, detalle=nombre,
                    critica=normalizar_nombre(nombre) in CATEGORIAS_CRITICAS))
    if faltantes:
        print(f"⚠️  {faltantes} imágenes del JSON no están en disco (¿descarga incompleta?).")
    return muestras, sin_mapeo


def muestras_carpeta(ruta: Path, solo_prueba: bool) -> list[Muestra]:
    """
    Crea muestras de imagen completa a partir de subcarpetas por clase.

    Cubre TrashNet y fotos propias (p. ej. `icopor/`, `paquete_papas/`): el nombre de
    la subcarpeta se resuelve con `mapear_categoria`.

    Args:
        ruta: Carpeta raíz.
        solo_prueba: TrashNet: conserva solo el 15 % de prueba que la red no vio
            (misma partición y semilla que `entrenamiento/entrenar_trashnet.py`).

    Returns:
        Lista de muestras sin caja (bbox=None).
    """
    archivos = sorted(p for p in ruta.rglob("*") if p.suffix.lower() in EXTENSIONES)
    if solo_prueba:
        archivos = _particion_prueba_trashnet(ruta)
    return [Muestra(archivo=p.relative_to(ruta).as_posix(), ruta=p, bbox=None, tam_anotado=None,
                    clase=mapear_categoria(p.parent.name), detalle=p.parent.name,
                    critica=normalizar_nombre(p.parent.name) in CATEGORIAS_CRITICAS)
            for p in archivos]


def _particion_prueba_trashnet(ruta: Path) -> list[Path]:
    """Reproduce el split 70/15/15 estratificado (semilla 42) del entrenamiento."""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        sys.exit("❌ --solo-prueba requiere scikit-learn:  pip install scikit-learn")
    clases = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    rutas, y = [], []
    for k, clase in enumerate(clases):
        archivos = sorted((ruta / clase).glob("*.jpg"))
        rutas += archivos
        y += [k] * len(archivos)
    if not rutas:
        sys.exit(f"❌ {ruta} no tiene la estructura de TrashNet.")
    y = np.array(y)
    _, tmp = train_test_split(np.arange(len(rutas)), test_size=0.30, stratify=y, random_state=42)
    _, prueba = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)
    return [rutas[int(i)] for i in sorted(prueba)]


# ===========================================================================
# 3. SIMULACIÓN DE WEBCAM Y AUMENTO DE DATOS (albumentations)
# ===========================================================================
LADO_WEBCAM = 640


def simular_camara(bgr: np.ndarray, calidad_jpeg: int = 75) -> np.ndarray:
    """
    Degradación determinista de la variante «original»: 640 px y compresión JPEG.

    Reproduce el frame que entrega WebRTC (VP8 comprimido) sin efectos aleatorios,
    para que la línea base sea comparable entre ejecuciones.

    Args:
        bgr: Imagen BGR.
        calidad_jpeg: Calidad de compresión (0-100).

    Returns:
        Imagen BGR degradada.
    """
    alto, ancho = bgr.shape[:2]
    escala = LADO_WEBCAM / max(alto, ancho)
    if escala < 1:
        bgr = cv2.resize(bgr, (int(ancho * escala), int(alto * escala)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, calidad_jpeg])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else bgr


def construir_aumentador(alto: int, ancho: int, *, metalizada: bool, semilla: int,
                         fraccion_recorte: float = 0.88) -> A.Compose:
    """
    Construye la tubería de albumentations para un recorte de tamaño (alto, ancho).

    Se crea por muestra porque `RandomCrop` exige un tamaño fijo en píxeles y los
    objetos COCO varían de 20 a 4000 px; así el recorte es siempre proporcional.
    La semilla derivada del archivo hace cada variante reproducible.

    Transformaciones (en orden físico de la cadena de captura):
        1. LongestMaxSize       Resolución de webcam (640 px).
        2. RandomCrop           Objeto descentrado / encuadre imperfecto.
        3. RandomSunFlare       Reflejo especular de lámparas sobre BOPP metalizado
                                y latas (p=0.8 en clases metalizadas, 0.3 resto).
        4. RandomBrightnessContrast  Autoexposición y luz fluorescente de cocina.
        5. GaussNoise           Ruido de sensor CMOS en baja luz.
        6. MotionBlur           Mano que mueve el objeto frente a la cámara.
        7. ImageCompression     Artefactos de bloque del códec WebRTC.

    Args:
        alto: Alto del recorte de entrada.
        ancho: Ancho del recorte de entrada.
        metalizada: True para empaques/superficies reflectivas.
        semilla: Semilla del generador aleatorio de la tubería.
        fraccion_recorte: Fracción del lado que conserva `RandomCrop`.

    Returns:
        `A.Compose` listo para llamar con `aumentador(image=bgr)["image"]`.
    """
    escala = min(1.0, LADO_WEBCAM / max(alto, ancho))
    alto_r, ancho_r = max(1, round(alto * escala)), max(1, round(ancho * escala))
    alto_c, ancho_c = max(8, int(alto_r * fraccion_recorte)), max(8, int(ancho_r * fraccion_recorte))
    radio = max(8, int(0.30 * min(alto_c, ancho_c)))
    return A.Compose([
        A.LongestMaxSize(max_size=LADO_WEBCAM, p=1.0) if escala < 1 else A.NoOp(),
        A.RandomCrop(height=min(alto_c, alto_r), width=min(ancho_c, ancho_r), p=1.0),
        A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 1.0), src_radius=radio,
                         num_flare_circles_range=(3, 7), angle_range=(0.0, 1.0),
                         method="overlay", p=0.8 if metalizada else 0.3),
        A.RandomBrightnessContrast(brightness_limit=(-0.30, 0.25), contrast_limit=(-0.25, 0.20),
                                   p=0.9),
        A.GaussNoise(std_range=(0.02, 0.07), p=0.4),
        A.MotionBlur(blur_limit=(3, 7), p=0.25),
        A.ImageCompression(compression_type="jpeg", quality_range=(55, 85), p=1.0),
    ], seed=semilla)


def semilla_muestra(semilla_global: int, archivo: str, variante: int) -> int:
    """Semilla estable por (ejecución, archivo, variante) independiente del orden."""
    return zlib.crc32(f"{semilla_global}|{archivo}|{variante}".encode()) & 0x7FFFFFFF


def generar_variantes(muestra: Muestra, base: np.ndarray, *, aumentos: int,
                      incluir_original: bool, semilla: int,
                      calidad_jpeg: int) -> Iterator[tuple[int, np.ndarray]]:
    """
    Produce las imágenes a evaluar para una muestra.

    Args:
        muestra: Muestra de origen (aporta la clase para los destellos).
        base: Recorte BGR del objeto con margen.
        aumentos: Número de variantes aumentadas.
        incluir_original: Si True, emite primero la variante 0 (degradación determinista).
        semilla: Semilla global de la ejecución.
        calidad_jpeg: Calidad de la variante original.

    Yields:
        (índice de variante, imagen BGR). 0 = original, 1..N = aumentadas.
    """
    if incluir_original:
        yield 0, simular_camara(base, calidad_jpeg)
    metalizada = bool(muestra.clase and muestra.clase.metalizada)
    for k in range(1, aumentos + 1):
        aumentador = construir_aumentador(*base.shape[:2], metalizada=metalizada,
                                          semilla=semilla_muestra(semilla, muestra.archivo, k))
        yield k, aumentador(image=base)["image"]


# ===========================================================================
# 4. EVALUACIÓN, REGISTRO Y REPORTES
# ===========================================================================
def fechas_simuladas(dias: int, rng: random.Random) -> Iterator[datetime | None]:
    """
    Genera fechas aleatorias en horario 7:00-19:00 de los últimos `dias` días.

    Args:
        dias: Ventana en días; 0 deja que la BD use la hora actual.
        rng: Generador aleatorio reproducible.

    Yields:
        Fecha simulada (nunca futura) o None.
    """
    if dias <= 0:
        while True:
            yield None
    inicio = datetime.now().replace(hour=7, minute=0, second=0, microsecond=0) \
        - timedelta(days=dias - 1)
    while True:
        f = inicio + timedelta(days=rng.randrange(dias), seconds=rng.randrange(12 * 3600))
        yield min(f, datetime.now())


def barra_progreso(i: int, total: int, t0: float, guardados: int) -> None:
    """Imprime una barra de progreso en una sola línea de consola."""
    pct = i / max(1, total)
    vel = i / max(1e-6, time.perf_counter() - t0)
    resta = (total - i) / max(vel, 1e-6)
    barra = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))
    print(f"\r  {barra} {pct:5.1%}  {i}/{total} objetos  {vel:5.1f} obj/s  "
          f"guardables: {guardados}  ~{resta:4.0f}s ", end="", flush=True)


@dataclass
class Estadisticas:
    """
    Acumuladores de validación por variante (original vs. aumentada).

    Attributes:
        confusion: variante -> clase normativa -> Counter(grupo predicho).
        criticas: variante -> (categoría, grupo esperado) -> Counter(grupo predicho).
    """

    confusion: dict[str, dict[str, Counter]]
    criticas: dict[str, dict[tuple[str, str], Counter]]

    @classmethod
    def nueva(cls) -> "Estadisticas":
        """Crea acumuladores vacíos."""
        return cls(defaultdict(lambda: defaultdict(Counter)),
                   defaultdict(lambda: defaultdict(Counter)))

    def agregar(self, variante: str, muestra: Muestra, grupo_predicho: str) -> None:
        """Suma una predicción con verdad de terreno evaluable."""
        self.confusion[variante][muestra.clase.nombre][grupo_predicho] += 1
        if muestra.critica:
            self.criticas[variante][(muestra.detalle, muestra.clase.grupo)][grupo_predicho] += 1


def ejecutar(args: argparse.Namespace) -> int:
    """
    Orquesta ingesta -> mapeo -> aumento -> evaluación -> registro.

    Args:
        args: Argumentos de línea de comandos (ver `main`).

    Returns:
        Código de salida del proceso (0 = éxito).
    """
    ruta = Path(args.ruta).expanduser().resolve()
    if not ruta.is_dir():
        sys.exit(f"❌ No existe la carpeta: {ruta}")
    rng = random.Random(args.semilla)

    rutas_json = [Path(p).resolve() for p in args.anotaciones] if args.anotaciones \
        else buscar_anotaciones(ruta)
    if rutas_json:
        tipo = "taco" if (ruta / "annotations.json").is_file() and not args.anotaciones else "coco"
        muestras, sin_mapeo = muestras_coco(
            rutas_json, ruta, lado_minimo=args.lado_minimo,
            max_por_imagen=args.max_por_imagen, incluir_sin_mapeo=args.incluir_sin_mapeo)
    else:
        tipo = "trashnet" if (ruta / "cardboard").is_dir() and (ruta / "trash").is_dir() else "carpeta"
        muestras = muestras_carpeta(ruta, args.solo_prueba)
        sin_mapeo = Counter(m.detalle for m in muestras if m.clase is None)
    if sin_mapeo:
        lista = ", ".join(f"{k} ({v})" for k, v in sin_mapeo.most_common(12))
        print(f"⚠️  Categorías sin clase normativa"
              f"{' (procesadas sin etiqueta)' if args.incluir_sin_mapeo else ' (omitidas)'}: {lista}"
              "\n    Agréguelas a MAPEO_CATEGORIAS o use --incluir-sin-mapeo.")

    if args.solo_criticas:
        muestras = [m for m in muestras if m.critica]
    rng.shuffle(muestras)                         # Muestreo aleatorio reproducible
    if args.limite:
        muestras = muestras[:args.limite]
    if not muestras:
        sys.exit("❌ No hay muestras para procesar con los filtros indicados.")
    muestras.sort(key=lambda m: str(m.ruta))       # Agrupa por foto -> aprovecha la caché

    origen = args.origen or tipo
    origen_aug = f"{origen}{SUFIJO_AUMENTADO}"
    variantes = args.aumentos + (0 if args.sin_original else 1)
    print(f"📂 Dataset: {tipo} ({len(rutas_json)} JSON)  |  objetos: {len(muestras)}  |  "
          f"variantes/objeto: {variantes}  ->  {len(muestras) * variantes} evaluaciones")
    print(f"🗄️  Origen en BD: '{origen}' (original) / '{origen_aug}' (aumentadas)  |  "
          f"BD: {db.RUTA_BD}{'  |  DRY-RUN' if args.dry_run else ''}")
    print(f"🧠 Motor: {args.motor}  |  umbral de registro: >{args.umbral:.0f}%"
          f"{'  |  guardando TODO' if args.guardar_todo else ''}")

    clasificador = ClasificadorResiduos()
    if clasificador.red_onnx is None and args.motor in MOTORES[:2]:
        print(f"⚠️  Red neuronal no disponible ({clasificador.error_carga_dnn}).")

    exportar = Path(args.exportar_recortes).resolve() if args.exportar_recortes else None
    fechas = fechas_simuladas(args.distribuir_dias, rng)
    stats = Estadisticas.nueva()
    lote: list[dict] = []
    filas_csv: list[dict] = []
    insertados = guardables = errores = exportados = 0

    def volcar() -> None:
        """Escribe el lote pendiente en UNA transacción (WAL: no bloquea al tablero)."""
        nonlocal insertados
        if lote and not args.dry_run:
            insertados += db.crear_lote(lote)
        lote.clear()

    t0 = time.perf_counter()
    for i, m in enumerate(muestras, start=1):
        try:
            base = recortar(leer_imagen(m.ruta, m.tam_anotado), m.bbox, args.margen)
            for k, imagen in generar_variantes(m, base, aumentos=args.aumentos,
                                               incluir_original=not args.sin_original,
                                               semilla=args.semilla,
                                               calidad_jpeg=args.calidad_jpeg):
                pred = clasificador.predecir(imagen, motor=args.motor)
                variante = "original" if k == 0 else "aumentada"
                id_archivo = m.archivo if k == 0 else f"{m.archivo}@aug{k}"
                evaluable = m.clase is not None and m.clase.evaluable
                grupo_pred = db.grupo_material(pred.material)
                if evaluable:
                    stats.agregar(variante, m, grupo_pred)

                pasa = pred.confianza > args.umbral and pred.material != "No identificado"
                if pasa or args.guardar_todo:
                    guardables += 1
                    lote.append({
                        "material": pred.material, "confianza": min(100.0, pred.confianza),
                        "contenedor": pred.contenedor,
                        "origen": origen if k == 0 else origen_aug, "motor": args.motor,
                        "etiqueta_real": m.clase.nombre if evaluable else None,
                        "archivo": id_archivo, "fecha_hora": next(fechas),
                    })
                    if len(lote) >= args.lote:
                        volcar()
                if exportar and m.clase and m.clase.clase_trashnet:
                    destino = exportar / m.clase.clase_trashnet
                    destino.mkdir(parents=True, exist_ok=True)
                    nombre = re.sub(r"[^\w.-]+", "_", id_archivo) + ".jpg"
                    exportados += cv2.imwrite(str(destino / nombre), imagen)
                if args.reporte_csv:
                    filas_csv.append({
                        "archivo": id_archivo, "variante": variante, "categoria": m.detalle,
                        "critica": m.critica, "clase_normativa": m.clase.nombre if m.clase else "",
                        "contenedor_esperado": (m.clase.contenedor or "") if m.clase else "",
                        "material_predicho": pred.material, "grupo_predicho": grupo_pred,
                        "acierto": (grupo_pred == m.clase.grupo) if evaluable else "",
                        "confianza": pred.confianza, "contenedor_predicho": pred.contenedor,
                        "registrado": pasa or args.guardar_todo})
        except Exception as exc:  # Una imagen corrupta no debe detener la ingesta masiva
            errores += 1
            if args.verbose:
                print(f"\n  ✗ {m.archivo}: {type(exc).__name__}: {exc}")
        if i % 5 == 0 or i == len(muestras):
            barra_progreso(i, len(muestras), t0, guardables)
    volcar()
    duracion = time.perf_counter() - t0
    print()

    imprimir_resumen(len(muestras), guardables, insertados, errores, duracion, args)
    for variante in ("original", "aumentada"):
        if stats.confusion.get(variante):
            imprimir_validacion(variante, stats.confusion[variante], stats.criticas.get(variante, {}))
    if exportar:
        print(f"\n🖼️  {exportados} recortes exportados en {exportar} (estructura TrashNet): "
              "combínelos con TrashNet y ejecute entrenamiento/entrenar_trashnet.py <carpeta> "
              "<backbone.onnx> <salida.onnx> para reentrenar.")
    if args.reporte_csv and filas_csv:
        with open(args.reporte_csv, "w", newline="", encoding="utf-8-sig") as f:
            escritor = csv.DictWriter(f, fieldnames=list(filas_csv[0]))
            escritor.writeheader()
            escritor.writerows(filas_csv)
        print(f"📝 Detalle por evaluación: {args.reporte_csv}")
    return 0


def imprimir_resumen(total: int, guardables: int, insertados: int, errores: int,
                     duracion: float, args: argparse.Namespace) -> None:
    """Imprime rendimiento y conteo de inserciones en SQLite."""
    print("\n" + "=" * 78)
    print(f"✅ Objetos procesados: {total - errores}/{total} en {duracion:.1f}s "
          f"({(total - errores) / max(duracion, 1e-6):.1f} obj/s)  |  errores: {errores}"
          f"{'' if args.verbose or not errores else '  (use -v para el detalle)'}")
    if args.dry_run:
        print(f"💾 DRY-RUN: se habrían guardado {guardables} registros.")
    else:
        dup = guardables - insertados
        print(f"💾 Insertados en SQLite: {insertados}"
              + (f"  (omitidos {dup} ya existentes para este origen y motor)" if dup else ""))


def imprimir_validacion(variante: str, confusion: dict[str, Counter],
                        criticas: dict[tuple[str, str], Counter]) -> None:
    """
    Imprime exactitud, matriz de confusión por clase normativa y recall crítico.

    La comparación se hace por grupo (`database.grupo_material`): el motor no
    distingue «Multicapa» de «Icopor», pero sí debe enviarlos a «No aprovechable».

    Args:
        variante: «original» o «aumentada».
        confusion: clase normativa -> Counter(grupo predicho).
        criticas: (categoría crítica, grupo esperado) -> Counter(grupo predicho).
    """
    n = sum(sum(c.values()) for c in confusion.values())
    aciertos = sum(c[db.grupo_material(real)] for real, c in confusion.items())
    no_id = sum(c["No identificado"] for c in confusion.values())
    print(f"\n🧪 VALIDACIÓN — variante {variante.upper()} ({n} evaluaciones, sin aplicar umbral)")
    print(f"   Exactitud global: {aciertos / n:6.1%}   |   "
          f"sobre identificados: {aciertos / max(1, n - no_id):6.1%}   |   "
          f"No identificado: {no_id / n:6.1%}")

    columnas = sorted({p for c in confusion.values() for p in c})
    ancho = max(10, *(len(c) for c in columnas))
    print("\n   Matriz de confusión (filas = clase normativa real, columnas = grupo predicho)")
    print("   " + " " * 30 + "".join(f"{c[:ancho]:>{ancho + 1}}" for c in columnas) + "   recall")
    for real in sorted(confusion):
        fila, tot = confusion[real], sum(confusion[real].values())
        print(f"   {real[:30]:<30}" + "".join(f"{fila[c]:>{ancho + 1}}" for c in columnas)
              + f"   {fila[db.grupo_material(real)] / tot:6.1%}")

    if criticas:
        print("\n   🎯 Brecha de dominio — categorías críticas")
        for (cat, esperado), fila in sorted(criticas.items(), key=lambda kv: -sum(kv[1].values())):
            tot = sum(fila.values())
            frecuente, cuantos = fila.most_common(1)[0]
            print(f"   {cat[:28]:<28} n={tot:<5} recall {fila[esperado] / tot:6.1%}   "
                  f"error más frecuente: {frecuente if frecuente != esperado else '—'}"
                  f"{f' ({cuantos / tot:.0%})' if frecuente != esperado else ''}")



def _consola_utf8() -> None:
    """
    Fuerza UTF-8 en la consola. En Windows, cuando la salida se redirige a un archivo
    o la ejecuta otra herramienta, Python usa cp1252 y falla con tildes o emojis.
    """
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    """
    Punto de entrada de la línea de comandos.

    Args:
        argv: Argumentos (None = `sys.argv[1:]`).

    Returns:
        Código de salida del proceso.
    """
    _consola_utf8()
    p = argparse.ArgumentParser(
        description="Ingesta COCO/TACO/Roboflow con aumento de datos hacia la BD de reciclaje.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ruta", help="Carpeta del dataset descargado.")
    p.add_argument("--anotaciones", nargs="+",
                   help="Archivos COCO explícitos (por defecto se buscan en --ruta).")
    p.add_argument("--motor", choices=MOTORES, default="Motor Híbrido Estricto")
    p.add_argument("--origen", help="Etiqueta 'origen' en la BD (por defecto el tipo de dataset).")
    p.add_argument("--limite", type=int, default=0, help="Máximo de objetos (0 = todos).")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--solo-criticas", action="store_true",
                   help="Solo multicapa, icopor, aerosoles y envases de aseo.")
    p.add_argument("--incluir-sin-mapeo", action="store_true",
                   help="Procesa categorías sin clase normativa (sin etiqueta real).")
    g = p.add_argument_group("aumento de datos")
    g.add_argument("--aumentos", type=int, default=2, help="Variantes aumentadas por objeto.")
    g.add_argument("--sin-original", action="store_true",
                   help="No evalúa la variante original (solo aumentadas).")
    g.add_argument("--calidad-jpeg", type=int, default=75, help="Calidad de la variante original.")
    g.add_argument("--margen", type=float, default=0.30, help="Contexto alrededor del bbox.")
    g.add_argument("--lado-minimo", type=int, default=48, help="Ignora objetos más pequeños (px).")
    g.add_argument("--max-por-imagen", type=int, default=3, help="Objetos por fotografía.")
    g.add_argument("--exportar-recortes",
                   help="Guarda las imágenes evaluadas en <dir>/<clase_trashnet>/ para reentrenar.")
    r = p.add_argument_group("registro")
    r.add_argument("--umbral", type=float, default=db.UMBRAL_REGISTRO,
                   help="Solo se guarda si confianza > umbral (misma política que la cámara).")
    r.add_argument("--guardar-todo", action="store_true",
                   help="Guarda también lo que no supera el umbral (validación sin sesgo).")
    r.add_argument("--distribuir-dias", type=int, default=0,
                   help="Reparte fecha_hora en los últimos N días (0 = hora actual).")
    r.add_argument("--lote", type=int, default=250, help="Filas por transacción SQLite.")
    r.add_argument("--dry-run", action="store_true", help="Evalúa y reporta sin escribir.")
    r.add_argument("--reporte-csv", help="Guarda el resultado por evaluación en este CSV.")
    r.add_argument("--borrar-origen", help="Elimina todos los registros de ese origen y termina.")
    p.add_argument("--solo-prueba", action="store_true",
                   help="TrashNet: usa solo el 15%% de prueba que el modelo no vio.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.borrar_origen:
        print(f"🗑️  Eliminados {db.eliminar_por_origen(args.borrar_origen)} registros "
              f"de '{args.borrar_origen}'.")
        return 0
    if not args.ruta:
        p.error("--ruta es obligatorio (o use --borrar-origen).")
    if args.aumentos < 0 or (args.aumentos == 0 and args.sin_original):
        p.error("Debe evaluarse al menos una variante por objeto.")
    return ejecutar(args)


if __name__ == "__main__":
    sys.exit(main())
