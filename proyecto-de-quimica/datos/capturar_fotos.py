"""
datos/capturar_fotos.py — Toma de fotos propias con la webcam (el dominio REAL del sistema).

Guarda en  fotos_propias/<sesion>/<clase>/<clase>_<fecha>_<n>.jpg , la estructura que lee
`python datos/construir_manifiesto.py --propias fotos_propias`.

    python datos/capturar_fotos.py --sesion cocina_tarde
    python datos/capturar_fotos.py --sesion prueba_patio --camara 1     # "prueba_*" = solo examen

Teclas en la ventana:
    0-9, a-f   elegir clase (0 = pet_botella … f = fondo_sin_objeto; la lista se ve en pantalla)
    ESPACIO    guardar una foto de la clase activa
    r          ráfaga de 10 fotos (mueva/gire el objeto mientras tanto)
    q / ESC    salir

Requiere una ventana de OpenCV: si tiene instalado `opencv-python-headless` (el de la app),
instale además el paquete con interfaz SOLO para esta herramienta:  pip install opencv-python

Recomendación: 50–100 fotos por clase, en 3–4 sesiones (lugar / hora / luz distintos).
Haga al menos UNA sesión llamada "prueba_..." que nunca se use para entrenar.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from taxonomia import Taxonomia  # noqa: E402

TECLAS = "0123456789abcdef"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Captura de fotos propias por clase.")
    p.add_argument("--sesion", required=True, help="Lugar/hora, p. ej. cocina_tarde (prueba_* = examen)")
    p.add_argument("--camara", type=int, default=0)
    p.add_argument("--destino", type=Path, default=RAIZ / "fotos_propias")
    p.add_argument("--ancho", type=int, default=1280)
    args = p.parse_args(argv)

    clases = Taxonomia.cargar().nombres
    carpeta = args.destino / args.sesion
    cam = cv2.VideoCapture(args.camara)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
    if not cam.isOpened():
        sys.exit(f"❌ No se pudo abrir la cámara {args.camara}")
    activa, rafaga, aviso, t_aviso = 0, 0, "", 0.0

    def contar(c: str) -> int:
        return len(list((carpeta / c).glob("*.jpg")))

    def guardar(frame) -> str:
        d = carpeta / clases[activa]
        d.mkdir(parents=True, exist_ok=True)
        nombre = f"{clases[activa]}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(d / nombre), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return nombre

    try:
        cv2.namedWindow("prueba")
        cv2.destroyWindow("prueba")
    except cv2.error:
        sys.exit("❌ Esta instalación de OpenCV no tiene ventanas (headless). "
                 "Ejecute: pip install opencv-python")
    print(f"Sesión '{args.sesion}' → {carpeta}")
    while True:
        ok, frame = cam.read()
        if not ok:
            print("⚠️  La cámara dejó de entregar imágenes.")
            break
        if rafaga:
            guardar(frame)
            rafaga -= 1
            aviso, t_aviso = f"Ráfaga: quedan {rafaga}", time.monotonic()
            time.sleep(0.25)
        vista = frame.copy()
        for i, c in enumerate(clases):
            color = (0, 220, 0) if i == activa else (230, 230, 230)
            cv2.putText(vista, f"{TECLAS[i]}: {c} ({contar(c)})", (10, 22 + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(vista, f"{TECLAS[i]}: {c} ({contar(c)})", (10, 22 + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if aviso and time.monotonic() - t_aviso < 1.5:
            cv2.putText(vista, aviso, (10, vista.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
        cv2.imshow("Captura de fotos propias  (ESPACIO guardar · r rafaga · q salir)", vista)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k < 128 and chr(k) in TECLAS[: len(clases)]:
            activa = TECLAS.index(chr(k))
        elif k == ord(" "):
            aviso, t_aviso = f"Guardada: {guardar(frame)}", time.monotonic()
        elif k == ord("r"):
            rafaga = 10
    cam.release()
    cv2.destroyAllWindows()
    print("Fotos por clase:", {c: contar(c) for c in clases if contar(c)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
