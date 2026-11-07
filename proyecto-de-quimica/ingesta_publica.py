"""
ingesta_publica.py — Ingesta y validación con bancos de imágenes públicos
=========================================================================
Recorre un dataset descargado localmente (TrashNet o TACO), pasa cada imagen por
el MISMO motor de visión que usa la cámara (simulando la captura de una webcam)
y guarda los resultados en SQLite para poblar el Tablero Estadístico.

Datasets soportados (detección automática):
    * TrashNet  (Thung & Yang, 2016 — MIT)
        <ruta>/cardboard/*.jpg, glass/, metal/, paper/, plastic/, trash/
        https://github.com/garythung/trashnet  -> data/dataset-resized.zip
    * TACO — Trash Annotations in Context (Proença & Simões, 2020 — CC BY 4.0)
        <ruta>/annotations.json  +  <ruta>/batch_1/*.jpg ... batch_15/
        https://github.com/pedropro/TACO  -> python download.py
        Cada anotación (bbox) se recorta como si el objeto estuviera frente a la cámara.
    * Carpeta genérica: imágenes sueltas o en subcarpetas (sin etiqueta real).

Ejemplos:
    python ingesta_publica.py --ruta D:/datasets/dataset-resized --limite 600
    python ingesta_publica.py --ruta D:/datasets/dataset-resized --solo-prueba
    python ingesta_publica.py --ruta D:/datasets/TACO/data --limite 800 --distribuir-dias 30
    python ingesta_publica.py --ruta D:/datasets/TACO/data --dry-run --reporte-csv taco.csv
    python ingesta_publica.py --borrar-origen taco          # deshacer una ingesta

⚠️ Fuga de datos: la red MobileNetV2 del proyecto se ENTRENÓ con TrashNet. Validar con
TrashNet completo da una exactitud inflada; usa --solo-prueba (el 15 % reservado, misma
semilla que entrenamiento/entrenar_trashnet.py) o, mejor aún, TACO, que el modelo nunca vio.

Autor: Johan Lasso
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
from PIL import Image, ImageOps

import database as db
from vision_engine import ClasificadorResiduos

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MOTORES = ["Motor Híbrido Estricto", "Motor IA (MobileNetV2)",
           "Motor Geométrico (Forma/Contornos)", "Motor Básico (Color)"]

# ---------------------------------------------------------------------------
# Etiquetas reales -> grupos del sistema (ver db.GRUPO_VALIDACION)
# ---------------------------------------------------------------------------
CLASES_TRASHNET = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
ETIQUETA_TRASHNET = {
    "cardboard": "Papel/Cartón",
    "paper": "Papel/Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "plastic": "Plástico",
    "trash": "No aprovechable",
}

# Las 60 categorías oficiales de TACO. None = sin equivalente comparable
# (orgánico, peligroso o sin etiqueta): se clasifica pero no cuenta para exactitud.
ETIQUETA_TACO = {
    "Aluminium foil": "Metal",
    "Battery": None,                         # Residuo peligroso (posconsumo)
    "Aluminium blister pack": "No aprovechable",
    "Carded blister pack": "No aprovechable",
    "Other plastic bottle": "Plástico",
    "Clear plastic bottle": "Plástico",
    "Glass bottle": "Vidrio",
    "Plastic bottle cap": "Plástico",
    "Metal bottle cap": "Metal",
    "Broken glass": "Vidrio",
    "Food Can": "Metal",
    "Aerosol": "Metal",
    "Drink can": "Metal",
    "Toilet tube": "Papel/Cartón",
    "Other carton": "Papel/Cartón",
    "Egg carton": "Papel/Cartón",
    "Drink carton": "Papel/Cartón",
    "Corrugated carton": "Papel/Cartón",
    "Meal carton": "Papel/Cartón",
    "Pizza box": "Papel/Cartón",
    "Paper cup": "No aprovechable",          # Papel plastificado
    "Disposable plastic cup": "Plástico",
    "Foam cup": "No aprovechable",
    "Glass cup": "Vidrio",
    "Other plastic cup": "Plástico",
    "Food waste": None,                      # Orgánico: el modelo no tiene esa clase
    "Glass jar": "Vidrio",
    "Plastic lid": "Plástico",
    "Metal lid": "Metal",
    "Other plastic": "Plástico",
    "Magazine paper": "Papel/Cartón",
    "Tissues": "No aprovechable",
    "Wrapping paper": "Papel/Cartón",
    "Normal paper": "Papel/Cartón",
    "Paper bag": "Papel/Cartón",
    "Plastified paper bag": "No aprovechable",
    "Plastic film": "No aprovechable",
    "Six pack rings": "No aprovechable",
    "Garbage bag": "No aprovechable",
    "Other plastic wrapper": "No aprovechable",
    "Single-use carrier bag": "No aprovechable",
    "Polypropylene bag": "No aprovechable",
    "Crisp packet": "No aprovechable",
    "Spread tub": "Plástico",
    "Tupperware": "Plástico",
    "Disposable food container": "Plástico",
    "Foam food container": "No aprovechable",
    "Other plastic container": "Plástico",
    "Plastic glooves": "No aprovechable",
    "Plastic utensils": "No aprovechable",
    "Pop tab": "Metal",
    "Rope & strings": "No aprovechable",
    "Scrap metal": "Metal",
    "Shoe": "No aprovechable",
    "Squeezable tube": "No aprovechable",
    "Plastic straw": "No aprovechable",
    "Paper straw": "No aprovechable",
    "Styrofoam piece": "No aprovechable",
    "Unlabeled litter": None,
    "Cigarette": "No aprovechable",
}


# ===========================================================================
# 1. MUESTRAS
# ===========================================================================
@dataclass
class Muestra:
    archivo: str                                  # Id estable (ruta relativa [+ #anotación])
    cargar: Callable[[], np.ndarray]              # Devuelve BGR al momento de procesar (lazy)
    etiqueta_real: str | None
    detalle: str = ""                             # Clase original del dataset


def leer_imagen(ruta: Path) -> np.ndarray:
    """
    Lee con PIL (soporta rutas con tildes en Windows, a diferencia de cv2.imread)
    y aplica la orientación EXIF. Imprescindible en TACO: muchas fotos de celular
    vienen rotadas y las anotaciones están sobre la imagen ya orientada.
    """
    with Image.open(ruta) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def detectar_dataset(ruta: Path) -> str:
    if (ruta / "annotations.json").is_file():
        return "taco"
    subcarpetas = {p.name.lower() for p in ruta.iterdir() if p.is_dir()}
    if len(subcarpetas & set(CLASES_TRASHNET)) >= 4:
        return "trashnet"
    return "carpeta"


def muestras_trashnet(ruta: Path, solo_prueba: bool) -> list[Muestra]:
    rutas, etiquetas = [], []
    for k, clase in enumerate(CLASES_TRASHNET):
        archivos = sorted((ruta / clase).glob("*.jpg"))   # Mismo orden que el entrenamiento
        rutas += archivos
        etiquetas += [k] * len(archivos)
    if not rutas:
        sys.exit(f"❌ No se encontraron imágenes TrashNet en {ruta}")

    indices = range(len(rutas))
    if solo_prueba:
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            sys.exit("❌ --solo-prueba requiere scikit-learn:  pip install scikit-learn")
        y = np.array(etiquetas)
        _, tmp = train_test_split(np.arange(len(rutas)), test_size=0.30, stratify=y, random_state=42)
        _, prueba = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)
        indices = sorted(int(i) for i in prueba)

    return [Muestra(archivo=rutas[i].relative_to(ruta).as_posix(),
                    cargar=(lambda p=rutas[i]: leer_imagen(p)),
                    etiqueta_real=ETIQUETA_TRASHNET[CLASES_TRASHNET[etiquetas[i]]],
                    detalle=CLASES_TRASHNET[etiquetas[i]])
            for i in indices]


def muestras_taco(ruta: Path, margen: float, lado_minimo: int, max_por_imagen: int) -> list[Muestra]:
    datos = json.loads((ruta / "annotations.json").read_text(encoding="utf-8"))
    imagenes = {im["id"]: im for im in datos["images"]}
    categorias = {c["id"]: c["name"] for c in datos["categories"]}

    por_imagen: dict[int, list[dict]] = defaultdict(list)
    for an in datos["annotations"]:
        _, _, w, h = an["bbox"]
        if min(w, h) >= lado_minimo:
            por_imagen[an["image_id"]].append(an)

    muestras, faltantes = [], 0
    for id_img, anotaciones in por_imagen.items():
        meta = imagenes[id_img]
        archivo_img = ruta / meta["file_name"]
        if not archivo_img.is_file():
            faltantes += 1
            continue
        # Prioriza los objetos más grandes (los que una persona acercaría a la cámara)
        anotaciones = sorted(anotaciones, key=lambda a: a["area"], reverse=True)[:max_por_imagen]
        for an in anotaciones:
            nombre = categorias[an["category_id"]]
            muestras.append(Muestra(
                archivo=f"{meta['file_name']}#{an['id']}",
                cargar=(lambda p=archivo_img, b=tuple(an["bbox"]): _recortar(leer_imagen(p), b, margen)),
                etiqueta_real=ETIQUETA_TACO.get(nombre),
                detalle=nombre,
            ))
    if faltantes:
        print(f"⚠️  {faltantes} imágenes de annotations.json no están en disco "
              f"(¿descarga incompleta? ejecuta TACO/download.py).")
    if not muestras:
        sys.exit(f"❌ No hay imágenes TACO utilizables en {ruta}")
    return muestras


def _recortar(bgr: np.ndarray, bbox: tuple[float, float, float, float], margen: float) -> np.ndarray:
    alto, ancho = bgr.shape[:2]
    x, y, w, h = bbox
    px, py = w * margen, h * margen
    x1, y1 = max(0, int(x - px)), max(0, int(y - py))
    x2, y2 = min(ancho, int(x + w + px)), min(alto, int(y + h + py))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError(f"bbox fuera de la imagen: {bbox}")
    return bgr[y1:y2, x1:x2]


def muestras_carpeta(ruta: Path) -> list[Muestra]:
    archivos = sorted(p for p in ruta.rglob("*") if p.suffix.lower() in EXTENSIONES)
    if not archivos:
        sys.exit(f"❌ No se encontraron imágenes en {ruta}")
    return [Muestra(archivo=p.relative_to(ruta).as_posix(), cargar=(lambda p=p: leer_imagen(p)),
                    etiqueta_real=ETIQUETA_TRASHNET.get(p.parent.name.lower()),
                    detalle=p.parent.name) for p in archivos]


# ===========================================================================
# 2. SIMULACIÓN DE CÁMARA
# ===========================================================================
def simular_camara(bgr: np.ndarray, calidad_jpeg: int, lado: int = 640) -> np.ndarray:
    """
    Reproduce las condiciones de un frame de webcam: resolución 640 px en el lado
    mayor y artefactos de compresión (el frame WebRTC llega comprimido en VP8).
    """
    alto, ancho = bgr.shape[:2]
    escala = lado / max(alto, ancho)
    if escala < 1:
        bgr = cv2.resize(bgr, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, calidad_jpeg])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else bgr


# ===========================================================================
# 3. PROCESO PRINCIPAL
# ===========================================================================
def fechas_simuladas(n: int, dias: int, rng: random.Random) -> Iterator[datetime | None]:
    """Reparte los registros en horario 7:00-19:00 de los últimos `dias` días."""
    if dias <= 0:
        while True:
            yield None
    inicio = datetime.now().replace(hour=7, minute=0, second=0, microsecond=0) - timedelta(days=dias - 1)
    fechas = sorted(inicio + timedelta(days=rng.randrange(dias), seconds=rng.randrange(12 * 3600))
                    for _ in range(n))
    yield from (min(f, datetime.now()) for f in fechas)


def barra_progreso(i: int, total: int, t0: float, ok: int) -> None:
    pct = i / total
    vel = i / max(1e-6, time.perf_counter() - t0)
    restante = (total - i) / max(vel, 1e-6)
    print(f"\r  [{'█' * int(30 * pct):<30}] {i}/{total}  {vel:5.1f} img/s  "
          f"guardables: {ok}  ETA {restante:4.0f}s", end="", flush=True)


def ejecutar(args: argparse.Namespace) -> int:
    ruta = Path(args.ruta).expanduser().resolve()
    if not ruta.is_dir():
        sys.exit(f"❌ La ruta no existe o no es una carpeta: {ruta}")

    tipo = detectar_dataset(ruta) if args.dataset == "auto" else args.dataset
    origen = args.origen or (f"{tipo}-prueba" if tipo == "trashnet" and args.solo_prueba else tipo)
    rng = random.Random(args.semilla)

    if tipo == "trashnet":
        muestras = muestras_trashnet(ruta, args.solo_prueba)
    elif tipo == "taco":
        muestras = muestras_taco(ruta, args.margen, args.lado_minimo, args.max_por_imagen)
    else:
        muestras = muestras_carpeta(ruta)

    rng.shuffle(muestras)                 # Muestreo aleatorio reproducible (todas las clases)
    if args.limite:
        muestras = muestras[:args.limite]

    print(f"📂 Dataset: {tipo}  |  origen en BD: '{origen}'  |  muestras: {len(muestras)}")
    print(f"🧠 Motor: {args.motor}  |  umbral de registro: >{args.umbral:.0f}%"
          f"{'  |  guardando TODO' if args.guardar_todo else ''}"
          f"{'  |  DRY-RUN (sin escribir)' if args.dry_run else ''}")
    if tipo == "trashnet" and not args.solo_prueba:
        print("⚠️  El modelo se entrenó con TrashNet: la exactitud estará inflada. "
              "Usa --solo-prueba o TACO para una validación honesta.")

    clasificador = ClasificadorResiduos()
    if clasificador.red_onnx is None and args.motor in MOTORES[:2]:
        print(f"⚠️  Red neuronal no disponible ({clasificador.error_carga_dnn}); "
              "los resultados serán 'No identificado' o del motor geométrico.")

    fechas = fechas_simuladas(len(muestras), args.distribuir_dias, rng)
    lote: list[dict] = []
    insertados = guardables = errores = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    filas_csv: list[dict] = []

    def volcar() -> None:
        nonlocal insertados
        if lote and not args.dry_run:
            insertados += db.crear_lote(lote)
        lote.clear()

    t0 = time.perf_counter()
    for i, m in enumerate(muestras, start=1):
        fecha = next(fechas)
        try:
            bgr = m.cargar()
            if not args.sin_simulacion:
                bgr = simular_camara(bgr, args.calidad_jpeg)
            pred = clasificador.predecir(bgr, motor=args.motor)
        except Exception as exc:
            errores += 1
            if args.verbose:
                print(f"\n  ✗ {m.archivo}: {exc}")
            continue

        pasa_umbral = pred.confianza > args.umbral and pred.material != "No identificado"
        if m.etiqueta_real:
            confusion[m.etiqueta_real][db.grupo_material(pred.material)] += 1
        if args.reporte_csv:
            filas_csv.append({"archivo": m.archivo, "clase_dataset": m.detalle,
                              "etiqueta_real": m.etiqueta_real or "", "material": pred.material,
                              "grupo_predicho": db.grupo_material(pred.material),
                              "confianza": pred.confianza, "contenedor": pred.contenedor,
                              "registrado": pasa_umbral or args.guardar_todo})
        if pasa_umbral or args.guardar_todo:
            guardables += 1
            lote.append({"material": pred.material, "confianza": min(100.0, pred.confianza),
                         "contenedor": pred.contenedor, "origen": origen, "motor": args.motor,
                         "etiqueta_real": m.etiqueta_real, "archivo": m.archivo, "fecha_hora": fecha})
            if len(lote) >= args.lote:
                volcar()
        if i % 10 == 0 or i == len(muestras):
            barra_progreso(i, len(muestras), t0, guardables)
    volcar()
    duracion = time.perf_counter() - t0
    print()

    imprimir_reporte(confusion, len(muestras), guardables, insertados, errores, duracion, args)
    if args.reporte_csv and filas_csv:
        with open(args.reporte_csv, "w", newline="", encoding="utf-8-sig") as f:
            escritor = csv.DictWriter(f, fieldnames=list(filas_csv[0]))
            escritor.writeheader()
            escritor.writerows(filas_csv)
        print(f"📝 Detalle por imagen: {args.reporte_csv}")
    return 0


def imprimir_reporte(confusion: dict[str, Counter], total: int, guardables: int, insertados: int,
                     errores: int, duracion: float, args: argparse.Namespace) -> None:
    print("\n" + "=" * 72)
    print(f"✅ Procesadas: {total - errores}/{total} en {duracion:.1f}s "
          f"({(total - errores) / max(duracion, 1e-6):.1f} img/s)  |  errores: {errores}")
    if args.dry_run:
        print(f"💾 DRY-RUN: se habrían guardado {guardables} registros.")
    else:
        dup = guardables - insertados
        print(f"💾 Insertados en SQLite: {insertados}"
              + (f"  (omitidos {dup} ya existentes para este origen y motor)" if dup else ""))

    n = sum(sum(c.values()) for c in confusion.values())
    if not n:
        print("ℹ️  Sin etiquetas reales: no se calcula exactitud.")
        return
    no_id = sum(c["No identificado"] for c in confusion.values())
    aciertos = sum(c[real] for real, c in confusion.items())
    decididos = n - no_id
    print(f"\n🧪 VALIDACIÓN (todas las predicciones con etiqueta real, sin aplicar umbral): {n}")
    print(f"   Exactitud global:              {aciertos / n:6.1%}")
    print(f"   Exactitud sobre identificados: {aciertos / max(1, decididos):6.1%}")
    print(f"   No identificado (fallback):    {no_id / n:6.1%}")

    columnas = sorted({p for c in confusion.values() for p in c})
    ancho = max(14, *(len(c) for c in columnas))
    print("\n   Matriz de confusión (filas = real, columnas = predicho)")
    print("   " + " " * 16 + "".join(f"{c[:ancho]:>{ancho + 1}}" for c in columnas) + "   recall")
    for real in sorted(confusion):
        fila = confusion[real]
        tot = sum(fila.values())
        print(f"   {real[:16]:<16}" + "".join(f"{fila[c]:>{ancho + 1}}" for c in columnas)
              + f"   {fila[real] / tot:6.1%}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingesta de TrashNet / TACO en la BD de reciclaje.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ruta", help="Carpeta del dataset descargado.")
    p.add_argument("--dataset", choices=["auto", "trashnet", "taco", "carpeta"], default="auto")
    p.add_argument("--motor", choices=MOTORES, default="Motor Híbrido Estricto")
    p.add_argument("--limite", type=int, default=0, help="Máximo de muestras (0 = todas).")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--origen", help="Etiqueta 'origen' en la BD (por defecto el tipo de dataset).")
    p.add_argument("--umbral", type=float, default=db.UMBRAL_REGISTRO,
                   help="Solo se guarda si confianza > umbral (misma política que la cámara).")
    p.add_argument("--guardar-todo", action="store_true", help="Guarda también lo que no supera el umbral.")
    p.add_argument("--solo-prueba", action="store_true",
                   help="TrashNet: usa solo el 15%% de prueba que el modelo no vio al entrenar.")
    p.add_argument("--distribuir-dias", type=int, default=0,
                   help="Reparte fecha_hora en los últimos N días (0 = hora actual).")
    p.add_argument("--sin-simulacion", action="store_true", help="No degradar a calidad webcam.")
    p.add_argument("--calidad-jpeg", type=int, default=75)
    p.add_argument("--margen", type=float, default=0.15, help="TACO: margen alrededor del bbox.")
    p.add_argument("--lado-minimo", type=int, default=64, help="TACO: ignora objetos más pequeños (px).")
    p.add_argument("--max-por-imagen", type=int, default=3, help="TACO: objetos por foto.")
    p.add_argument("--lote", type=int, default=200, help="Filas por transacción SQLite.")
    p.add_argument("--dry-run", action="store_true", help="Clasifica y reporta sin escribir en la BD.")
    p.add_argument("--reporte-csv", help="Guarda el resultado por imagen en este CSV.")
    p.add_argument("--borrar-origen", help="Elimina todos los registros de ese origen y termina.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.borrar_origen:
        print(f"🗑️  Eliminados {db.eliminar_por_origen(args.borrar_origen)} registros "
              f"de '{args.borrar_origen}'.")
        return 0
    if not args.ruta:
        p.error("--ruta es obligatorio (o usa --borrar-origen).")
    return ejecutar(args)


if __name__ == "__main__":
    sys.exit(main())
