"""
entrenar_trashnet.py — Entrena el clasificador de residuos (transfer learning).

Técnica: Transfer Learning con "feature extraction".
    1. MobileNetV2 preentrenada en ImageNet (1,2 M de imágenes) actúa como
       extractor de características: convierte cada foto en un vector de 1280
       números que describe texturas, bordes, brillos y formas.
    2. Sobre esos vectores se entrena una capa final (regresión logística
       multinomial) con las fotos de TrashNet (6 clases).
    3. La capa entrenada se "pega" al final de la red y se exporta todo como
       un único archivo ONNX que la aplicación ejecuta con cv2.dnn, sin
       internet y sin TensorFlow/PyTorch.

Dataset: TrashNet (Gary Thung & Mindy Yang, Stanford 2016, licencia MIT)
         https://github.com/garythung/trashnet  -> data/dataset-resized.zip

Uso:
    pip install onnx onnxruntime scikit-learn opencv-python-headless numpy
    python entrenar_trashnet.py <carpeta_dataset> <backbone.onnx> <salida.onnx>
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CLASES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]  # Orden alfabético
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESV = np.array([0.229, 0.224, 0.225], np.float32)
SEMILLA = 42


def preprocesar(bgr):
    """Mismo preprocesamiento que vision_mock.ModeloTrashNet.predecir()."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    return ((rgb - MEDIA) / DESV).transpose(2, 0, 1)[None].astype(np.float32)


def aumentar(bgr, rng):
    """Variaciones realistas de una foto de webcam (aumentan la robustez)."""
    h, w = bgr.shape[:2]
    variantes = [bgr, cv2.flip(bgr, 1)]                       # Original y espejo
    escala = rng.uniform(0.70, 0.85)                          # Recorte (objeto más cerca)
    ch, cw = int(h * escala), int(w * escala)
    y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    variantes.append(bgr[y0:y0 + ch, x0:x0 + cw])
    alpha, beta = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)  # Iluminación distinta
    variantes.append(cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-25, 25), 1.0)  # Objeto girado
    variantes.append(cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return variantes


def extraer(sesion, imagenes):
    return np.vstack([sesion.run(None, {"input": preprocesar(im)})[0] for im in imagenes])


def main(carpeta, ruta_backbone, ruta_salida):
    rng = np.random.default_rng(SEMILLA)
    rutas, etiquetas = [], []
    for k, clase in enumerate(CLASES):
        archivos = sorted((Path(carpeta) / clase).glob("*.jpg"))
        rutas += archivos
        etiquetas += [k] * len(archivos)
        print(f"  {clase:10s} {len(archivos)} fotos")
    etiquetas = np.array(etiquetas)

    # División estratificada: 70 % entrenamiento, 15 % validación, 15 % prueba
    idx = np.arange(len(rutas))
    i_ent, i_tmp = train_test_split(idx, test_size=0.30, stratify=etiquetas, random_state=SEMILLA)
    i_val, i_pru = train_test_split(i_tmp, test_size=0.50, stratify=etiquetas[i_tmp], random_state=SEMILLA)

    opciones = ort.SessionOptions()
    sesion = ort.InferenceSession(ruta_backbone, opciones)
    t0 = time.time()
    X_ent, y_ent = [], []
    for n, i in enumerate(i_ent):
        variantes = aumentar(cv2.imread(str(rutas[i])), rng)
        X_ent.append(extraer(sesion, variantes))
        y_ent += [etiquetas[i]] * len(variantes)
        if n % 300 == 0:
            print(f"  características: {n}/{len(i_ent)} ({time.time() - t0:.0f}s)")
    X_ent, y_ent = np.vstack(X_ent), np.array(y_ent)
    X_val = extraer(sesion, [cv2.imread(str(rutas[i])) for i in i_val])
    X_pru = extraer(sesion, [cv2.imread(str(rutas[i])) for i in i_pru])
    y_val, y_pru = etiquetas[i_val], etiquetas[i_pru]

    # Estandarización (se pliega luego dentro de la capa final del ONNX)
    mu, sd = X_ent.mean(0), X_ent.std(0) + 1e-6
    norm = lambda X: (X - mu) / sd

    # Búsqueda del hiperparámetro de regularización C con el set de validación
    mejor = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(norm(X_ent), y_ent)
        acc = clf.score(norm(X_val), y_val)
        print(f"  C={C:<5} exactitud validación = {acc:.3f}")
        if mejor is None or acc > mejor[0]:
            mejor = (acc, C, clf)
    acc_val, C, clf = mejor

    # Evaluación final en el set de PRUEBA (fotos nunca vistas)
    pred = clf.predict(norm(X_pru))
    acc_pru = float((pred == y_pru).mean())
    reporte = classification_report(y_pru, pred, target_names=CLASES, digits=3)
    matriz = confusion_matrix(y_pru, pred)
    print(f"\nC elegido = {C} | exactitud en PRUEBA = {acc_pru:.3f}\n")
    print(reporte)
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print("           " + " ".join(f"{c[:6]:>6s}" for c in CLASES))
    for c, fila in zip(CLASES, matriz):
        print(f"{c:10s} " + " ".join(f"{v:6d}" for v in fila))

    # Plegado: logits = W·((f - mu)/sd) + b  =  (W/sd)·f + (b - W·mu/sd)
    W = clf.coef_ / sd
    b = clf.intercept_ - (clf.coef_ * (mu / sd)).sum(1)

    # Se agrega la capa final (Gemm) al grafo del extractor y se guarda un solo ONNX
    modelo = onnx.load(ruta_backbone)
    g = modelo.graph
    g.initializer.extend([numpy_helper.from_array(W.astype(np.float32), "cabeza_W"),
                          numpy_helper.from_array(b.astype(np.float32), "cabeza_b")])
    g.node.append(helper.make_node("Gemm", ["salida", "cabeza_W", "cabeza_b"], ["logits"], transB=1))
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, len(CLASES)]))
    onnx.checker.check_model(modelo)
    onnx.save(modelo, ruta_salida)

    metricas = {"exactitud_prueba": round(acc_pru, 4), "exactitud_validacion": round(acc_val, 4),
                "C": C, "n_entrenamiento": int(len(i_ent)), "n_validacion": int(len(i_val)),
                "n_prueba": int(len(i_pru)), "clases": CLASES, "matriz_confusion": matriz.tolist()}
    Path(ruta_salida).with_suffix(".json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    Path(ruta_salida).with_name("reporte_entrenamiento.txt").write_text(
        f"Exactitud en prueba: {acc_pru:.3f}\n\n{reporte}", encoding="utf-8")
    print(f"\nModelo guardado en {ruta_salida}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
