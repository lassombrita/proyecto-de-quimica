"""
motor_dos_etapas.py — Motor de PRODUCCIÓN: YOLO11n (1 clase) + MobileNetV3-Large (16 clases).

Carga los pesos que genera `entrenamiento/train_pipeline.py`:
    models/detector_residuo.onnx      + detector_residuo.json
    models/clasificador_material.onnx + clasificador_material.json  (logits ya divididos por T)
y decide la caneca con `taxonomia/canecas.yaml` (Etapa B, sin re-entrenar).

    from motor_dos_etapas import MotorDosEtapas
    m = MotorDosEtapas()
    m.detectar_y_clasificar(bgr)   # -> list[DeteccionMaterial]
    m.predecir(bgr)                # -> vision_engine.ResultadoClasificacion (objeto principal)

Backend: onnxruntime si está instalado (más rápido), si no cv2.dnn. Thread-safe (candado).

Prueba por consola:
    python motor_dos_etapas.py foto.jpg [otra.jpg ...]
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from taxonomia import Taxonomia

RAIZ = Path(__file__).resolve().parent
DIR_MODELOS = RAIZ / "models"

# Nombre legible para la UI / la base de datos
NOMBRES_UI: dict[str, str] = {
    "pet_botella": "PET (botella)",
    "pead_pp_envase": "PEAD/PP (envase rígido)",
    "plastico_rigido_otro": "Plástico rígido (otro)",
    "plastico_flexible_bolsa": "Plástico flexible / bolsa",
    "multicapa_flexible": "Multicapa flexible (snacks)",
    "multicapa_rigido": "Multicapa rígido (Tetra Pak)",
    "icopor": "Icopor",
    "vidrio": "Vidrio",
    "metal_lata": "Metal (lata)",
    "metal_otro": "Metal (otro)",
    "papel": "Papel",
    "carton": "Cartón",
    "sanitario_y_rechazo": "Sanitario / rechazo",
    "organico": "Orgánico",
    "peligroso": "Peligroso (posconsumo)",
    "fondo_sin_objeto": "Sin objeto",
}
# canecas.yaml -> valor que acepta la base de datos / la UI ("Blanco" | "Negro" | "Verde")
CANECA_A_CONTENEDOR = {"Blanca": "Blanco", "Negra": "Negro", "Verde": "Verde",
                       # Peligrosos NO van en ninguna caneca; la BD solo admite 3 colores, así que se
                       # registra como Negro y el motivo indica el punto posconsumo.
                       "Ninguna": "Negro"}


def contenedor_de_material_ui(nombre_ui: str, taxonomia: Taxonomia | None = None) -> str | None:
    """'Icopor' -> 'Negro' según canecas.yaml (para registrar votos estabilizados en la BD)."""
    tax = taxonomia or Taxonomia.cargar()
    for clase, nombre in NOMBRES_UI.items():
        if nombre == nombre_ui and clase in tax.reglas:
            return CANECA_A_CONTENEDOR.get(tax.caneca(clase) or "Negra", "Negro")
    return None


@dataclass(frozen=True)
class DeteccionMaterial:
    caja_px: tuple[int, int, int, int]       # x1, y1, x2, y2 en píxeles de la imagen original
    conf_detector: float
    clase: str | None                        # Clase de master.yaml (None = sin decisión)
    probabilidad: float                      # Probabilidad calibrada de la clase top-1
    contenedor: str                          # Blanco | Negro | Verde
    incierto: bool
    nota: str
    top3: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    @property
    def material_ui(self) -> str:
        if self.incierto:
            return "No identificado"
        return NOMBRES_UI.get(self.clase or "", self.clase or "No identificado")


class _Red:
    """Envoltura mínima: onnxruntime si existe, si no cv2.dnn."""

    def __init__(self, ruta: Path) -> None:
        self.backend = "cv2.dnn"
        try:
            import onnxruntime as ort
            opciones = ort.SessionOptions()
            opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._s = ort.InferenceSession(str(ruta), opciones, providers=["CPUExecutionProvider"])
            self._entrada = self._s.get_inputs()[0].name
            self.backend = "onnxruntime"
        except ImportError:
            self._s = None
            self._net = cv2.dnn.readNetFromONNX(str(ruta))

    def __call__(self, blob: np.ndarray) -> np.ndarray:
        if self._s is not None:
            return self._s.run(None, {self._entrada: blob})[0]
        self._net.setInput(blob)
        return self._net.forward()


def _letterbox(bgr: np.ndarray, lado: int) -> tuple[np.ndarray, float, int, int]:
    h, w = bgr.shape[:2]
    r = lado / max(h, w)
    nw, nh = round(w * r), round(h * r)
    lienzo = np.full((lado, lado, 3), 114, np.uint8)
    dx, dy = (lado - nw) // 2, (lado - nh) // 2
    lienzo[dy:dy + nh, dx:dx + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return lienzo, r, dx, dy


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class MotorDosEtapas:
    NOMBRE_MODELO = "YOLO11n (residuo) + MobileNetV3-Large (16 materiales)"

    def __init__(self, dir_modelos: Path = DIR_MODELOS, taxonomia: Taxonomia | None = None,
                 max_objetos: int = 5) -> None:
        self.dir_modelos = Path(dir_modelos)
        self.max_objetos = max_objetos
        self.error_carga: str | None = None
        self.detector = self.clasificador = None
        self._lock = threading.Lock()
        self.tax = taxonomia or Taxonomia.cargar()
        try:
            self.meta_det = json.loads((self.dir_modelos / "detector_residuo.json").read_text("utf-8"))
            self.meta_cls = json.loads((self.dir_modelos / "clasificador_material.json").read_text("utf-8"))
            self.detector = _Red(self.dir_modelos / "detector_residuo.onnx")
            self.clasificador = _Red(self.dir_modelos / "clasificador_material.onnx")
        except Exception as exc:  # noqa: BLE001 - la app muestra el motivo y usa el motor anterior
            self.error_carga = f"{type(exc).__name__}: {exc}"
            return
        self.clases: list[str] = list(self.meta_cls["clases"])
        if self.clases != list(self.tax.nombres):
            self.error_carga = ("Las clases de clasificador_material.json no coinciden con "
                                "taxonomia/master.yaml: re-entrene o restaure la taxonomía.")
            self.detector = self.clasificador = None
            return
        self.lado_det = int(self.meta_det.get("imgsz", 416))
        self.conf_det = float(self.meta_det.get("conf", 0.30))
        self.iou_det = float(self.meta_det.get("iou", 0.50))
        self.lado_cls = int(self.meta_cls.get("imgsz", 224))
        self.media = np.array(self.meta_cls.get("media", [0.485, 0.456, 0.406]), np.float32)
        self.desv = np.array(self.meta_cls.get("desv", [0.229, 0.224, 0.225]), np.float32)
        self.ms_ultima = 0.0

    @property
    def disponible(self) -> bool:
        return self.detector is not None and self.clasificador is not None

    # ------------------------------------------------------------- Etapa 1
    def detectar(self, bgr: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        img, r, dx, dy = _letterbox(bgr, self.lado_det)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)          # RGB, NCHW, [0,1]
        salida = np.asarray(self.detector(blob))[0]                          # (5, N): cx,cy,w,h,score
        if salida.shape[0] != 5 and salida.shape[1] == 5:
            salida = salida.T
        cx, cy, w, h, s = salida
        keep = s >= self.conf_det
        if not keep.any():
            return []
        cx, cy, w, h, s = cx[keep], cy[keep], w[keep], h[keep], s[keep]
        x1 = (cx - w / 2 - dx) / r
        y1 = (cy - h / 2 - dy) / r
        cajas = np.stack([x1, y1, w / r, h / r], 1)
        idx = cv2.dnn.NMSBoxes(cajas.tolist(), s.tolist(), self.conf_det, self.iou_det)
        H, W = bgr.shape[:2]
        res = []
        for i in np.array(idx).flatten()[: self.max_objetos]:
            x, y, ww, hh = cajas[i]
            c = (max(0, int(x)), max(0, int(y)), min(W, int(x + ww)), min(H, int(y + hh)))
            if c[2] - c[0] >= 8 and c[3] - c[1] >= 8:
                res.append((c, float(s[i])))
        return res

    # ------------------------------------------------------------- Etapa 2
    def _preparar(self, bgr: np.ndarray) -> np.ndarray:
        """Igual que la validación del pipeline: recorte -> lado corto 256 -> 255 -> centro 224."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        corto = int(self.lado_cls * 1.14)
        e = corto / min(h, w)
        rgb = cv2.resize(rgb, (max(corto, round(w * e)), max(corto, round(h * e))),
                         interpolation=cv2.INTER_AREA if e < 1 else cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
        y0, x0 = (h - self.lado_cls) // 2, (w - self.lado_cls) // 2
        rgb = rgb[y0:y0 + self.lado_cls, x0:x0 + self.lado_cls].astype(np.float32) / 255.0
        return ((rgb - self.media) / self.desv).transpose(2, 0, 1)

    def clasificar(self, recortes: list[np.ndarray]) -> np.ndarray:
        lote = np.stack([self._preparar(r) for r in recortes]).astype(np.float32)
        logits = np.concatenate([np.asarray(self.clasificador(lote[i:i + 1])) for i in range(len(lote))])
        return _softmax(logits)          # La temperatura ya va dentro del ONNX

    @staticmethod
    def _recorte(bgr: np.ndarray, caja, margen: float = 0.15) -> np.ndarray:
        x1, y1, x2, y2 = caja
        mx, my = int((x2 - x1) * margen), int((y2 - y1) * margen)
        H, W = bgr.shape[:2]
        return bgr[max(0, y1 - my):min(H, y2 + my), max(0, x1 - mx):min(W, x2 + mx)]

    # ------------------------------------------------------------- Pipeline
    def detectar_y_clasificar(self, bgr: np.ndarray, *, sin_detector_usar_imagen: bool = False,
                              sucio: bool | None = None) -> list[DeteccionMaterial]:
        if not self.disponible:
            raise RuntimeError(self.error_carga or "Motor de 2 etapas no disponible")
        t0 = time.perf_counter()
        with self._lock:
            cajas = self.detectar(bgr)
            if not cajas and sin_detector_usar_imagen:
                H, W = bgr.shape[:2]
                cajas = [((0, 0, W, H), 0.0)]
            if not cajas:
                self.ms_ultima = (time.perf_counter() - t0) * 1000
                return []
            probs = self.clasificar([self._recorte(bgr, c) for c, _ in cajas])
        salida = []
        for (caja, conf), p in zip(cajas, probs):
            dic = {c: float(v) for c, v in zip(self.clases, p)}
            top = sorted(dic.items(), key=lambda kv: kv[1], reverse=True)[:3]
            if top[0][0] == "fondo_sin_objeto":
                continue                                 # El detector se equivocó: no hay residuo
            d = self.tax.decidir(dic, sucio=sucio)
            salida.append(DeteccionMaterial(
                caja, conf, d.clase, d.confianza,
                CANECA_A_CONTENEDOR.get(d.caneca or "Negra", "Negro"), d.incierto,
                d.nota or ("Llevar a un punto posconsumo, NO a una caneca." if d.caneca == "Ninguna" else ""),
                tuple((c, round(v * 100, 1)) for c, v in top)))
        self.ms_ultima = (time.perf_counter() - t0) * 1000
        return salida

    def predecir(self, bgr: np.ndarray):
        """Compatibilidad con `ClasificadorResiduos.predecir`: objeto más grande -> ResultadoClasificacion."""
        from vision_engine import CATEGORIAS, ResultadoClasificacion
        dets = self.detectar_y_clasificar(bgr, sin_detector_usar_imagen=True)
        if not dets:
            info = CATEGORIAS["No identificado"]
            return ResultadoClasificacion("No identificado", 0.0, info["contenedor"],
                                          "No se detectó ningún residuo en la imagen.", "—",
                                          "Motor 2 Etapas", "Detector: sin objetos / fondo.")
        d = max(dets, key=lambda k: (k.caja_px[2] - k.caja_px[0]) * (k.caja_px[3] - k.caja_px[1]))
        anotada = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for k in dets:
            x1, y1, x2, y2 = k.caja_px
            cv2.rectangle(anotada, (x1, y1), (x2, y2), (22, 163, 74) if k is d else (150, 150, 150), 3)
        top = [(NOMBRES_UI.get(c, c), v) for c, v in d.top3]
        motivo = (f"Detector {d.conf_detector:.0%} · clase {d.clase} {d.probabilidad:.0%} "
                  f"(calibrada) · {self.ms_ultima:.0f} ms")
        if d.incierto:
            motivo = "Incertidumbre (umbral/margen de canecas.yaml) → Negra por precaución. " + motivo
        return ResultadoClasificacion(
            material=d.material_ui, confianza=round(d.probabilidad * 100, 1), contenedor=d.contenedor,
            descripcion=d.nota or self.tax.reglas.get(d.clase or "", {}).get("nota", ""),
            forma="—", metodo="Motor 2 Etapas", motivo=motivo, top_predicciones=top,
            imagen_anotada=anotada)


def main(argv: list[str]) -> int:  # pragma: no cover - utilidad de consola
    m = MotorDosEtapas()
    if not m.disponible:
        print("❌", m.error_carga)
        return 1
    print(f"Backend: {m.detector.backend}")
    for ruta in argv:
        bgr = cv2.imread(ruta)
        if bgr is None:
            print(f"{ruta}: no se pudo leer")
            continue
        dets = m.detectar_y_clasificar(bgr)
        print(f"{ruta}: {len(dets)} objeto(s) en {m.ms_ultima:.0f} ms")
        for d in dets:
            print(f"   {d.caja_px}  det {d.conf_detector:.2f}  → {d.material_ui:<30} "
                  f"{d.probabilidad:5.1%}  caneca {d.contenedor}  top3 {list(d.top3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
