"""
evaluar_sistema.py — Mide el desempeño del SISTEMA COMPLETO (forma + red + fallback)
sobre el 15 % de fotos de TrashNet reservadas para prueba (las mismas que
usa entrenar_trashnet.py, misma semilla). Ninguna de estas fotos se usó
para entrenar ni para ajustar umbrales.

Uso (desde la carpeta raíz del proyecto):
    python entrenamiento/evaluar_sistema.py <carpeta_dataset-resized>
"""
import collections
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vision_mock as vm  # noqa: E402

CLASES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
ESPERADO = {"cardboard": ("Papel/Cartón",), "paper": ("Papel/Cartón",), "glass": ("Vidrio",),
            "metal": ("Metal",), "plastic": ("Plástico", "PET", "PEAD"), "trash": ("No aprovechable",)}


def main(carpeta):
    rutas, etiquetas = [], []
    for k, c in enumerate(CLASES):
        archivos = sorted((Path(carpeta) / c).glob("*.jpg"))
        rutas += archivos
        etiquetas += [k] * len(archivos)
    etiquetas = np.array(etiquetas)
    idx = np.arange(len(rutas))
    _, tmp = train_test_split(idx, test_size=0.30, stratify=etiquetas, random_state=42)
    _, prueba = train_test_split(tmp, test_size=0.50, stratify=etiquetas[tmp], random_state=42)

    modelo = vm.ClasificadorResiduos()
    print(f"Motor: {modelo.NOMBRE_MODELO} | fotos de prueba: {len(prueba)}")
    aciertos = no_id = 0
    tabla = collections.defaultdict(collections.Counter)
    for i in prueba:
        clase = CLASES[etiquetas[i]]
        r = modelo.predecir(cv2.imread(str(rutas[i])))
        aciertos += r.material in ESPERADO[clase]
        no_id += r.material == "No identificado"
        tabla[clase][r.material] += 1
    n = len(prueba)
    decididos = n - no_id
    print(f"Identificados:                 {decididos / n:.1%}")
    print(f"Exactitud sobre identificados: {aciertos / max(1, decididos):.1%}")
    print(f"Errores (sobre el total):      {(decididos - aciertos) / n:.1%}")
    print(f"No identificado (fallback):    {no_id / n:.1%}")
    for c in CLASES:
        print(f"  {c:10s}", dict(tabla[c].most_common()))


if __name__ == "__main__":
    main(sys.argv[1])
