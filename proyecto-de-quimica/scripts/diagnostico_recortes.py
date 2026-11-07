"""
diagnostico_recortes.py — Verificación visual de los recortes de TACO
=====================================================================
Genera una cuadrícula con recortes REALES (los mismos que evalúa `ingesta_publica.py`)
y escribe la categoría del dataset sobre cada uno. Sirve para comprobar a simple
vista que las cajas COCO caen sobre el objeto correcto (sin desplazamiento ni
rotación por EXIF) antes de confiar en las métricas.

Uso (desde la carpeta del proyecto):
    python scripts/diagnostico_recortes.py --ruta ../TACO/data
    python scripts/diagnostico_recortes.py --ruta ../TACO/data --aumentados

Salida por defecto: docs/img/diagnostico_recortes.jpg (JPEG liviano, ~300 KB).

Autor: Johan Lasso
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import ingesta_publica as ip  # noqa: E402

GRUPOS_OBJETIVO = ["Plástico", "Papel/Cartón", "Vidrio", "Metal", "No aprovechable"]
CRITICAS = ["Crisp packet", "Foam food container", "Styrofoam piece", "Foam cup", "Aerosol"]


def elegir_muestras(muestras: list[ip.Muestra], por_grupo: int, rng: random.Random) -> list[ip.Muestra]:
    """
    Selecciona muestras variadas: `por_grupo` de cada grupo y una de cada categoría crítica.

    Args:
        muestras: Muestras COCO con clase normativa.
        por_grupo: Objetos por grupo de material.
        rng: Generador aleatorio reproducible.

    Returns:
        Lista de muestras a dibujar.
    """
    por_clave: dict[str, list[ip.Muestra]] = defaultdict(list)
    for m in muestras:
        if m.clase is None or not m.clase.evaluable:
            continue
        por_clave[m.clase.grupo].append(m)
        por_clave[m.detalle].append(m)
    elegidas: list[ip.Muestra] = []
    for g in GRUPOS_OBJETIVO:
        elegidas += rng.sample(por_clave[g], min(por_grupo, len(por_clave[g])))
    for c in CRITICAS:
        candidatas = [m for m in por_clave[c] if m not in elegidas]
        if candidatas:
            elegidas.append(rng.choice(candidatas))
    return elegidas


def rotular(img: np.ndarray, texto: str, lado: int) -> np.ndarray:
    """Redimensiona a un cuadrado de `lado` px (con bandas) y escribe el texto arriba."""
    alto, ancho = img.shape[:2]
    escala = lado / max(alto, ancho)
    img = cv2.resize(img, (max(1, int(ancho * escala)), max(1, int(alto * escala))),
                     interpolation=cv2.INTER_AREA)
    lienzo = np.full((lado + 28, lado, 3), 245, np.uint8)
    y0, x0 = 28 + (lado - img.shape[0]) // 2, (lado - img.shape[1]) // 2
    lienzo[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    cv2.putText(lienzo, texto[:30], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1,
                cv2.LINE_AA)
    return lienzo


def main(argv: list[str] | None = None) -> int:
    """
    Punto de entrada.

    Args:
        argv: Argumentos de línea de comandos.

    Returns:
        Código de salida.
    """
    p = argparse.ArgumentParser(description="Cuadrícula de recortes TACO para verificación visual.")
    p.add_argument("--ruta", required=True, help="Carpeta con annotations.json (TACO/data).")
    p.add_argument("--salida", default=str(RAIZ / "docs" / "img" / "diagnostico_recortes.jpg"))
    p.add_argument("--por-grupo", type=int, default=2)
    p.add_argument("--aumentados", action="store_true",
                   help="Agrega al lado de cada recorte una variante aumentada (webcam).")
    p.add_argument("--margen", type=float, default=0.30)
    p.add_argument("--semilla", type=int, default=7)
    p.add_argument("--lado", type=int, default=220)
    args = p.parse_args(argv)

    ruta = Path(args.ruta).resolve()
    rutas_json = ip.buscar_anotaciones(ruta)
    if not rutas_json:
        sys.exit(f"No se encontró annotations.json en {ruta}")
    muestras, _ = ip.muestras_coco(rutas_json, ruta, lado_minimo=48, max_por_imagen=3,
                                   incluir_sin_mapeo=False)
    elegidas = elegir_muestras(muestras, args.por_grupo, random.Random(args.semilla))

    celdas = []
    for m in elegidas:
        base = ip.recortar(ip.leer_imagen(m.ruta, m.tam_anotado), m.bbox, args.margen)
        celdas.append(rotular(base, f"{m.detalle} -> {m.clase.grupo}", args.lado))
        if args.aumentados:
            _, aum = next(ip.generar_variantes(m, base, aumentos=1, incluir_original=False,
                                               semilla=args.semilla, calidad_jpeg=75))
            celdas.append(rotular(aum, "  (variante webcam)", args.lado))

    columnas = 4
    vacia = np.full_like(celdas[0], 255)
    celdas += [vacia] * (-len(celdas) % columnas)
    filas = [np.hstack(celdas[i:i + columnas]) for i in range(0, len(celdas), columnas)]
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida), np.vstack(filas), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"Cuadrícula con {len(elegidas)} objetos guardada en {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
