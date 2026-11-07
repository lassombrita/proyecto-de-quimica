"""
vision_edge.py — Procesamiento de video asíncrono para el Escáner en Vivo
========================================================================
Sistema de Reciclaje Inteligente · Pipeline Edge (pensado para NVIDIA Jetson)

Contiene:
    * TAXONOMIA               Clases del mercado colombiano -> contenedor (Res. 2184 de 2019).
    * ModeloVisionSimulado    Mock de inferencia (misma interfaz que tendrá el modelo real).
    * ProcesadorResiduosEdge  Clase heredada de `VideoProcessorBase` (streamlit-webrtc).

Arquitectura de hilos (por qué el video no acumula retraso):

    Navegador ──WebRTC──► recv()  [hilo de streamlit-webrtc, async_processing=True]
                            │  1. frame -> ndarray BGR (OpenCV)
                            │  2. copia reducida -> buzón queue.Queue(maxsize=1)
                            │     (si está lleno se DESCARTA el frame viejo: sin buffer bloat)
                            │  3. dibuja las últimas detecciones publicadas  (~1 ms)
                            ▼
                     buzón de 1 frame
                            │
                            ▼
            _bucle_inferencia()  [hilo daemon propio, uno por sesión de video]
                            │  preprocesar() -> modelo.inferir() -> detecciones
                            ├─► publica detecciones (tupla inmutable, swap atómico bajo lock)
                            └─► RegistradorAsincrono.encolar()  (no bloqueante)
                                         │
                                         ▼
                              hilo "sqlite-escritor" (database.py) -> SQLite WAL

Regla de oro: `recv()` NUNCA espera a la IA ni al disco. Si la inferencia tarda
80 ms y la cámara entrega 30 fps, los frames intermedios se descartan en el
buzón en vez de encolarse; la caja dibujada siempre corresponde al frame más
reciente analizado y la latencia percibida se mantiene constante.

Autor: Johan Lasso
"""

from __future__ import annotations

import threading
import time
import unicodedata
import zlib
from dataclasses import dataclass, field
from queue import Empty, Full, Queue

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from database import RegistradorAsincrono, color_de_contenedor

# ===========================================================================
# 1. TAXONOMÍA (mercado colombiano · Resolución 2184 de 2019)
# ===========================================================================
DESTINO_BLANCO = "BLANCO (Aprovechable)"
DESTINO_NEGRO = "NEGRO (No Aprovechable - Material de Rechazo)"
DESTINO_VERDE = "VERDE (Orgánico)"


@dataclass(frozen=True, slots=True)
class ClaseResiduo:
    nombre: str
    destino: str

    @property
    def color(self) -> str:
        return color_de_contenedor(self.destino)


TAXONOMIA: tuple[ClaseResiduo, ...] = (
    ClaseResiduo("Envase PET / PEAD (Limpieza)", DESTINO_BLANCO),
    ClaseResiduo("Envase Multicapa (Tetra Pak)", DESTINO_BLANCO),
    ClaseResiduo("Metales / Vidrio / Fibras de Cartón", DESTINO_BLANCO),
    ClaseResiduo("Envoltura Flexible Multicapa (Paquetes Snacks)", DESTINO_NEGRO),
    ClaseResiduo("Poliestireno Expandido (Icopor)", DESTINO_NEGRO),
    ClaseResiduo("Biomasa / Restos Orgánicos", DESTINO_VERDE),
)

# Colores BGR para dibujar con OpenCV. El "blanco" se dibuja blanco con borde
# oscuro para que sea visible sobre fondos claros.
COLOR_CAJA_BGR: dict[str, tuple[int, int, int]] = {
    "BLANCO": (245, 245, 245),
    "NEGRO": (25, 25, 25),
    "VERDE": (74, 163, 22),
}
COLOR_TEXTO_BGR: dict[str, tuple[int, int, int]] = {
    "BLANCO": (20, 20, 20),
    "NEGRO": (255, 255, 255),
    "VERDE": (255, 255, 255),
}

CERTIDUMBRE_MIN, CERTIDUMBRE_MAX = 75.0, 99.0

# --- Parámetros del pipeline -------------------------------------------------
ANCHO_INFERENCIA = 320          # La IA trabaja sobre una copia reducida (menos CPU/GPU)
INTERVALO_MIN_INFERENCIA_S = 1 / 15   # Tope de ~15 inferencias/s: deja CPU al códec VP8
VIGENCIA_DETECCION_S = 1.0      # Detecciones más viejas no se dibujan (IA detenida / lenta)
AREA_MIN_OBJETO = 0.02          # El objeto debe ocupar al menos el 2 % del encuadre


# ===========================================================================
# 2. ESTRUCTURAS INMUTABLES COMPARTIDAS ENTRE HILOS
# ===========================================================================
@dataclass(frozen=True, slots=True)
class Deteccion:
    """
    Resultado de inferencia. Inmutable: el hilo de IA crea una tupla nueva de
    detecciones y la "publica" reemplazando una referencia; el hilo de video
    solo lee esa referencia. Nadie modifica un objeto que otro hilo esté usando.

    `caja` va NORMALIZADA (0-1) para poder dibujarla sobre el frame original
    aunque la inferencia se haya hecho sobre una copia reducida.
    """
    material: str
    contenedor: str
    certidumbre: float
    caja: tuple[float, float, float, float]   # x1, y1, x2, y2 en [0, 1]
    marca_tiempo: float                       # time.monotonic() de la inferencia

    @property
    def color(self) -> str:
        return color_de_contenedor(self.contenedor)


@dataclass(frozen=True, slots=True)
class MetricasVideo:
    fps_video: float = 0.0
    fps_inferencia: float = 0.0
    latencia_ms: float = 0.0
    frames_recibidos: int = 0
    frames_descartados: int = 0
    inferencias: int = 0
    eventos_encolados: int = 0
    eventos_rechazados: int = 0
    ultima: Deteccion | None = None
    error: str | None = None


# ===========================================================================
# 3. TRANSFORMACIONES OPENCV
# ===========================================================================
def reducir(bgr: np.ndarray, ancho: int = ANCHO_INFERENCIA) -> np.ndarray:
    """Copia reducida (INTER_AREA evita aliasing). Siempre devuelve un arreglo NUEVO."""
    alto, ancho_orig = bgr.shape[:2]
    if ancho_orig <= ancho:
        return bgr.copy()
    escala = ancho / ancho_orig
    return cv2.resize(bgr, (ancho, max(1, int(alto * escala))), interpolation=cv2.INTER_AREA)


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def preprocesar(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transformaciones básicas previas a la inferencia.

    Returns:
        (bgr_normalizado, mascara_bordes)
        * Ecualización CLAHE sobre el canal L (LAB): compensa iluminación
          irregular sin alterar el tono.
        * Gris + desenfoque gaussiano + Canny adaptativo (umbral por mediana)
          + cierre morfológico: máscara de bordes para proponer la región del objeto.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    normalizado = cv2.cvtColor(cv2.merge((_CLAHE.apply(l), a, b)), cv2.COLOR_LAB2BGR)

    gris = cv2.GaussianBlur(cv2.cvtColor(normalizado, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # Umbrales de Canny derivados de Otsu: se adaptan al contraste objeto/fondo de
    # cada escena (la regla de la mediana falla cuando el fondo domina el encuadre).
    otsu, _ = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alto_c = float(np.clip(otsu, 40, 220))
    bordes = cv2.Canny(gris, 0.5 * alto_c, alto_c)
    mascara = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, _KERNEL, iterations=2)
    mascara = cv2.dilate(mascara, _KERNEL, iterations=1)
    return normalizado, mascara


def proponer_region(mascara: np.ndarray) -> tuple[int, int, int, int] | None:
    """Caja (x, y, w, h) del contorno externo más grande dentro de un rango de área."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alto, ancho = mascara.shape[:2]
    area_total = float(alto * ancho)
    candidatos = [c for c in contornos
                  if AREA_MIN_OBJETO * area_total <= cv2.contourArea(c) <= 0.9 * area_total]
    if not candidatos:
        return None
    return cv2.boundingRect(max(candidatos, key=cv2.contourArea))


# ===========================================================================
# 4. MODELO DE VISIÓN SIMULADO (Vision Model Mockup)
# ===========================================================================
class ModeloVisionSimulado:
    """
    Sustituto del modelo real con su MISMA interfaz: `inferir(bgr) -> list[Deteccion]`.

    Estrategia de simulación:
      * Región: REAL. La caja sale de contornos OpenCV (preprocesar + proponer_region),
        así el bounding box sigue al objeto que se pone frente a la cámara.
      * Clase: PSEUDOALEATORIA PERO ESTABLE. Se deriva de una firma cromática
        gruesa del recorte (histograma HSV cuantizado -> CRC32). El mismo objeto
        tiende a producir la misma clase de un frame a otro (no "parpadea"),
        y objetos distintos producen clases distintas.
      * Certidumbre: ESTOCÁSTICA en [75 %, 99 %]: base derivada de la firma +
        ruido gaussiano por inferencia.

    Para conectar el modelo real (TensorRT / ONNX en la Jetson) basta con otra
    clase que implemente `inferir()`; el procesador de video no cambia.

    Concurrencia: el mock no guarda estado mutable entre llamadas (el generador
    aleatorio lo aporta cada procesador), por lo que una única instancia cacheada
    con @st.cache_resource puede atender a varias sesiones a la vez. Un motor
    real (cv2.dnn.Net, contexto TensorRT) NO es thread-safe: ese caso se protege
    con el candado `self._lock`, que aquí se conserva como patrón.
    """

    nombre = "Mock estocástico v1 (taxonomía Res. 2184)"

    def __init__(self, taxonomia: tuple[ClaseResiduo, ...] = TAXONOMIA) -> None:
        self.taxonomia = taxonomia
        self._lock = threading.Lock()

    def inferir(self, bgr: np.ndarray, rng: np.random.Generator | None = None) -> list[Deteccion]:
        rng = rng if rng is not None else np.random.default_rng()
        normalizado, mascara = preprocesar(bgr)
        region = proponer_region(mascara)
        if region is None:
            return []
        x, y, w, h = region
        recorte = normalizado[y:y + h, x:x + w]

        with self._lock:   # Punto de entrada del motor real (no thread-safe)
            firma = self._firma(recorte)
        clase = self.taxonomia[firma % len(self.taxonomia)]
        base = CERTIDUMBRE_MIN + (firma >> 8) % 1000 / 1000 * (CERTIDUMBRE_MAX - CERTIDUMBRE_MIN - 4)
        certidumbre = float(np.clip(base + rng.normal(2.0, 1.5), CERTIDUMBRE_MIN, CERTIDUMBRE_MAX))

        alto, ancho = bgr.shape[:2]
        caja = (x / ancho, y / alto, (x + w) / ancho, (y + h) / alto)
        return [Deteccion(clase.nombre, clase.destino, round(certidumbre, 2), caja, time.monotonic())]

    @staticmethod
    def _firma(recorte: np.ndarray) -> int:
        """Histograma HSV de 4x2 cubetas (tono x saturación) -> entero estable."""
        hsv = cv2.cvtColor(cv2.resize(recorte, (32, 32), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [4, 2], [0, 180, 0, 256]).flatten()
        dominante = np.argsort(hist)[::-1][:2].astype(np.uint8)   # 2 cubetas dominantes
        return zlib.crc32(dominante.tobytes())


# ===========================================================================
# 5. DIBUJO DE BOUNDING BOXES
# ===========================================================================
def _ascii(texto: str) -> str:
    """cv2.putText solo dibuja ASCII: 'Cartón' -> 'Carton'."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def dibujar_detecciones(img: np.ndarray, detecciones: tuple[Deteccion, ...]) -> None:
    """Sobrepone cajas y etiquetas con el color del contenedor destino (in-place)."""
    alto, ancho = img.shape[:2]
    for d in detecciones:
        x1, y1 = int(d.caja[0] * ancho), int(d.caja[1] * alto)
        x2, y2 = int(d.caja[2] * ancho), int(d.caja[3] * alto)
        color = COLOR_CAJA_BGR[d.color]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 5, cv2.LINE_AA)   # contorno de contraste
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

        linea1 = f"{_ascii(d.material)}  {d.certidumbre:.1f}%"
        linea2 = f"-> {_ascii(d.contenedor)}"
        fuente, esc, grosor = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (w1, h1), _ = cv2.getTextSize(linea1, fuente, esc, grosor)
        (w2, _), _ = cv2.getTextSize(linea2, fuente, esc, grosor)
        alto_etq = 2 * h1 + 18
        ty = y1 - alto_etq if y1 - alto_etq >= 0 else min(y2, alto - alto_etq)
        cv2.rectangle(img, (x1, ty), (x1 + max(w1, w2) + 12, ty + alto_etq), color, -1)
        cv2.rectangle(img, (x1, ty), (x1 + max(w1, w2) + 12, ty + alto_etq), (0, 0, 0), 1)
        texto = COLOR_TEXTO_BGR[d.color]
        cv2.putText(img, linea1, (x1 + 6, ty + h1 + 5), fuente, esc, texto, grosor, cv2.LINE_AA)
        cv2.putText(img, linea2, (x1 + 6, ty + 2 * h1 + 12), fuente, esc, texto, grosor, cv2.LINE_AA)


def dibujar_hud(img: np.ndarray, m: MetricasVideo) -> None:
    """Franja superior con métricas de rendimiento (barata: solo 28 px)."""
    franja = img[0:28, :]
    franja[:] = (franja * 0.35).astype(np.uint8)
    texto = (f"Video {m.fps_video:4.1f} fps | IA {m.fps_inferencia:4.1f} inf/s | "
             f"{m.latencia_ms:5.1f} ms | descartados {m.frames_descartados}")
    cv2.putText(img, texto, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


# ===========================================================================
# 6. PROCESADOR DE VIDEO (streamlit-webrtc)
# ===========================================================================
class ProcesadorResiduosEdge(VideoProcessorBase):
    """
    Productor/consumidor desacoplado con buzón de capacidad 1.

    Hilos que tocan esta instancia y qué comparten:
      * Hilo de video (recv):   escribe el buzón; lee `_detecciones` y `_metricas`.
      * Hilo de inferencia:     lee el buzón; escribe `_detecciones` y `_metricas`.
      * Hilo del script (UI):   lee `metricas()`; ajusta parámetros de registro.

    `_detecciones` y `_metricas` son objetos inmutables que se reemplazan
    completos bajo `_lock`: las secciones críticas duran nanosegundos (swap de
    referencia) y ningún hilo puede ver un estado a medio actualizar.
    """

    def __init__(self, modelo: ModeloVisionSimulado, registrador: RegistradorAsincrono, *,
                 intervalo_registro_s: float = 0.5, umbral_registro: float = CERTIDUMBRE_MIN,
                 mostrar_hud: bool = True) -> None:
        self._modelo = modelo
        self._registrador = registrador
        # Parámetros ajustables en caliente desde la UI. Asignar un float/bool es
        # atómico en CPython (GIL), por lo que no requieren candado.
        self.intervalo_registro_s = float(intervalo_registro_s)
        self.umbral_registro = float(umbral_registro)
        self.mostrar_hud = mostrar_hud
        self.pausado = False

        self._buzon: Queue[np.ndarray] = Queue(maxsize=1)   # anti buffer bloat
        self._lock = threading.Lock()
        self._detecciones: tuple[Deteccion, ...] = ()
        self._metricas = MetricasVideo()

        self._rng = np.random.default_rng()      # RNG propio: no se comparte entre hilos
        self._t_frame_anterior: float | None = None
        self._fps_video = 0.0
        self._recibidos = 0
        self._descartados = 0

        self._detener = threading.Event()
        self._hilo = threading.Thread(target=self._bucle_inferencia,
                                      name="inferencia-edge", daemon=True)
        self._hilo.start()

    # ------------------------------------------------------------------ API UI
    def metricas(self) -> MetricasVideo:
        with self._lock:
            return self._metricas

    # ------------------------------------------------------- Hilo de video
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        imagen = frame.to_ndarray(format="bgr24")   # Tensor NumPy (alto, ancho, 3) uint8

        ahora = time.perf_counter()
        if self._t_frame_anterior is not None:
            dt = max(1e-3, ahora - self._t_frame_anterior)
            self._fps_video = 0.9 * self._fps_video + 0.1 / dt   # media móvil exponencial
        self._t_frame_anterior = ahora
        self._recibidos += 1

        if not self.pausado:
            # `reducir` devuelve una copia: el dibujo de abajo no contamina la entrada de la IA.
            self._depositar(reducir(imagen))

        with self._lock:
            detecciones, metricas = self._detecciones, self._metricas
        vigentes = tuple(d for d in detecciones
                         if time.monotonic() - d.marca_tiempo <= VIGENCIA_DETECCION_S)
        dibujar_detecciones(imagen, vigentes)
        if self.mostrar_hud:
            dibujar_hud(imagen, metricas)
        return av.VideoFrame.from_ndarray(imagen, format="bgr24")

    def _depositar(self, imagen: np.ndarray) -> None:
        """
        Política "último gana" sobre queue.Queue(maxsize=1).

        Si el buzón tiene un frame que la IA aún no tomó, ese frame ya es viejo:
        se retira y se deja el nuevo. Hay un único productor (este hilo), así que
        la única carrera posible es con el consumidor, que puede vaciar el buzón
        entre `get_nowait` y `put_nowait`; ambos casos se toleran sin bloquear.
        """
        try:
            self._buzon.put_nowait(imagen)
            return
        except Full:
            pass
        try:
            self._buzon.get_nowait()
            self._descartados += 1
        except Empty:
            pass
        try:
            self._buzon.put_nowait(imagen)
        except Full:
            self._descartados += 1

    def on_ended(self) -> None:
        """El usuario pulsó STOP o cerró la pestaña: se libera el hilo de inferencia."""
        self._detener.set()

    # -------------------------------------------------- Hilo de inferencia
    def _bucle_inferencia(self) -> None:
        latencia_ms = fps_inf = 0.0
        inferencias = encolados = rechazados = 0
        t_ultimo_registro = -float("inf")
        error: str | None = None

        while not self._detener.is_set():
            try:
                imagen = self._buzon.get(timeout=0.5)
            except Empty:
                continue   # Sin frames (pausa / red lenta): revisa _detener y sigue

            t0 = time.perf_counter()
            try:
                detecciones = tuple(self._modelo.inferir(imagen, self._rng))
                error = None
            except Exception as exc:     # El hilo nunca debe morir en silencio
                detecciones, error = (), f"Inferencia: {type(exc).__name__}: {exc}"
            latencia = time.perf_counter() - t0

            # Registro asíncrono, limitado por tasa y umbral (no bloqueante).
            ahora = time.monotonic()
            if detecciones and ahora - t_ultimo_registro >= self.intervalo_registro_s:
                for d in detecciones:
                    if d.certidumbre >= self.umbral_registro:
                        if self._registrador.encolar_deteccion(d.material, d.contenedor, d.certidumbre):
                            encolados += 1
                        else:
                            rechazados += 1
                t_ultimo_registro = ahora

            inferencias += 1
            ciclo = max(latencia, INTERVALO_MIN_INFERENCIA_S)
            latencia_ms = latencia * 1000 if inferencias == 1 else 0.8 * latencia_ms + 0.2 * latencia * 1000
            fps_inf = 1 / ciclo if inferencias == 1 else 0.8 * fps_inf + 0.2 / ciclo

            metricas = MetricasVideo(
                fps_video=self._fps_video, fps_inferencia=fps_inf, latencia_ms=latencia_ms,
                frames_recibidos=self._recibidos, frames_descartados=self._descartados,
                inferencias=inferencias, eventos_encolados=encolados, eventos_rechazados=rechazados,
                ultima=detecciones[0] if detecciones else self._metricas.ultima, error=error,
            )
            with self._lock:   # Publicación atómica (swap de referencias)
                self._detecciones = detecciones
                self._metricas = metricas

            # Tope de tasa: en la Jetson la CPU también decodifica/codifica VP8.
            resto = INTERVALO_MIN_INFERENCIA_S - (time.perf_counter() - t0)
            if resto > 0:
                self._detener.wait(resto)
