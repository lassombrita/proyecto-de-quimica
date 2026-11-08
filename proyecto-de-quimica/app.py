"""
app.py — Interfaz Gráfica Principal (Streamlit + WebRTC)
=========================================================
Reciclaje Inteligente — Clasificación de residuos en tiempo real y tablero analítico
Autor: Johan Lasso

Pantallas:
    1. "📷 Clasificador en Vivo": video continuo con `streamlit-webrtc`. La IA corre
       en un hilo secundario; la interfaz NO se recarga por cada fotograma.
    2. "📊 Tablero Estadístico": KPIs y gráficos Plotly alimentados por SQLite.

Arquitectura del flujo de video (por qué ya no hay lag):

    Navegador ──WebRTC──► recv() [hilo aiortc]          ──► frame + HUD ──► Navegador
                             │  (solo copia el último frame, ~1 ms)
                             ▼
                      buzón de 1 frame  (los viejos se descartan: nunca hay cola)
                             │
                             ▼
                  _bucle_inferencia() [hilo propio]  ── MobileNetV2 / OpenCV (~30-80 ms)
                             │
                             ├─► EstabilizadorDetecciones (votación en ventana)
                             │        └─► confianza > 60 % y estable ─► SQLite
                             ▼
                      estado compartido (lock) ◄── st.fragment(run_every=1 s)
                                                  (solo se re-renderiza el panel)

Ejecución:
    streamlit run app.py                       # solo motores de producción (IA / Híbrido)
    streamlit run app.py -- --modo-didactico   # agrega Color y Geometría para la exposición
    (equivalente: variable de entorno RECICLAJE_MODO_DIDACTICO=1)

Los eventos producidos por motores didácticos se guardan con origen
"camara_didactico", de modo que nunca se mezclan con las estadísticas reales.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import re
import threading
import time
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from statistics import fmean

import av
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

import database as db
from vision_engine import (
    CATEGORIAS,
    COLORES_CONTENEDOR,
    MOTORES_DIDACTICOS,
    MOTORES_PRODUCCION,
    ClasificadorResiduos,
)

# ===========================================================================
# 1. CONFIGURACIÓN
# ===========================================================================
st.set_page_config(
    page_title="Reciclaje Inteligente",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RUTA_CSS = Path(__file__).parent / "assets" / "styles.css"
PANTALLA_CLASIFICADOR = "📷 Clasificador en Vivo"
PANTALLA_TABLERO = "📊 Tablero Estadístico"
MODO_VIDEO = "🎥 Video en vivo (WebRTC)"
MODO_FOTO = "📸 Foto única (respaldo)"

# Modo didáctico: `streamlit run app.py -- --modo-didactico` o RECICLAJE_MODO_DIDACTICO=1.
MODO_DIDACTICO = ("--modo-didactico" in sys.argv[1:]
                  or os.environ.get("RECICLAJE_MODO_DIDACTICO", "").strip().lower() in ("1", "true", "si", "sí"))
MOTORES = list(MOTORES_PRODUCCION) + (list(MOTORES_DIDACTICOS) if MODO_DIDACTICO else [])
ORIGEN_CAMARA = "camara"
ORIGEN_DIDACTICO = "camara_didactico"


def origen_registro(motor: str) -> str:
    """Los eventos de motores didácticos no se mezclan con los de producción."""
    return ORIGEN_DIDACTICO if motor in MOTORES_DIDACTICOS else ORIGEN_CAMARA

# --- Parámetros del pipeline de video ---------------------------------------
UMBRAL_REGISTRO = db.UMBRAL_REGISTRO      # Se guarda solo si confianza > 60 %
ANCHO_INFERENCIA = 480                    # Igual a ANCHO_PROCESAMIENTO del motor
INTERVALO_INFERENCIA_S = 0.12             # Máx. ~8 inferencias/s (libera CPU para el códec)
VENTANA_ESTABILIDAD = 8                   # Últimas N inferencias consideradas
VOTOS_MINIMOS = 6                         # 6 de 8 deben coincidir en el material
INFERENCIAS_LIBERACION = 4                # Inferencias "sin objeto estable" para rearmar
INTERVALO_MIN_REGISTRO_S = 2.0            # Anti-rebote entre dos registros
# Zona de interés (fracciones x1, y1, x2, y2): el residuo se coloca dentro del recuadro.
ROI = (0.15, 0.08, 0.85, 0.92)
FRAMES_CALIBRACION = 12                   # ~1,5 s de fondo vacío al iniciar
UMBRAL_PRESENCIA = 0.04                   # 4 % de la zona con cambio = hay objeto

# Servidor STUN público: necesario cuando el navegador no está en la misma máquina.
# Para redes con NAT simétrico (universidades, 4G) hace falta además un servidor TURN.
RTC_CONFIGURACION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# 640x480 es suficiente: el motor reduce a 480 px y aiortc codifica VP8 por
# software; pedir 1280x720 cuadruplica el costo de (de)codificación por frame.
RESTRICCIONES_MEDIA = {
    "video": {
        "width": {"ideal": 640},
        "height": {"ideal": 480},
        "frameRate": {"ideal": 24, "max": 30},
    },
    "audio": False,
}

# --- Paleta del tablero (validada para daltonismo; color = tipo de residuo) ---
COLORES_TIPO = {
    "Aprovechable": "#2a78d6",      # azul
    "No aprovechable": "#eb6834",   # naranja
    "Orgánico": "#1baf7a",          # aqua
}
ORDEN_TIPOS = list(COLORES_TIPO)
COLOR_GRILLA = "rgba(15, 23, 42, 0.08)"


# ===========================================================================
# 2. RECURSOS COMPARTIDOS
# ===========================================================================
class MotorCompartido:
    """
    Envuelve al clasificador con un candado.

    `cv2.dnn.Net` NO es thread-safe: con WebRTC cada pestaña abierta tiene su
    propio hilo de inferencia y todos comparten la red cargada por
    @st.cache_resource. Sin este lock, dos `setInput/forward` simultáneos
    corrompen la salida o bloquean el proceso.
    """

    def __init__(self) -> None:
        self.clasificador = ClasificadorResiduos(modo_didactico=MODO_DIDACTICO)
        self._lock = threading.Lock()

    @property
    def nombre(self) -> str:
        return self.clasificador.NOMBRE_MODELO

    def predecir(self, imagen, motor: str):
        with self._lock:
            return self.clasificador.predecir(imagen, motor=motor)


@st.cache_resource(show_spinner="🧠 Cargando motor de Visión por Computador...")
def cargar_motor() -> MotorCompartido:
    db.init_db()
    return MotorCompartido()


@st.cache_data(show_spinner=False)
def leer_css(ruta: str) -> str:
    css = Path(ruta).read_text(encoding="utf-8")
    return re.sub(r"</\s*style\s*>", "", css, flags=re.IGNORECASE)


def cargar_css(ruta: Path) -> None:
    if ruta.exists():
        st.markdown(f"<style>{leer_css(str(ruta))}</style>", unsafe_allow_html=True)


# ===========================================================================
# 3. ESTABILIZADOR DE DETECCIONES (lógica pura, sin Streamlit)
# ===========================================================================
class EstabilizadorDetecciones:
    """
    Convierte un flujo ruidoso de predicciones por frame en EVENTOS discretos.

    Regla: se registra un residuo cuando, en las últimas `ventana` inferencias,
    el mismo material gana al menos `votos_minimos` veces y su confianza media
    supera el umbral (> 60 %). Luego el material queda "bloqueado" para no
    registrar el mismo objeto 8 veces por segundo; se rearma cuando la escena
    pierde estabilidad (se retiró el objeto) o aparece un material distinto.
    """

    NO_REGISTRABLES = {"No identificado"}

    def escena_vacia(self) -> None:
        """La zona quedó vacía: se descarta la votación y, tras `liberacion` lecturas, se rearma."""
        self._historial.clear()
        self._inestables += 1
        if self._inestables >= self.liberacion:
            self._bloqueado = None

    def __init__(self, ventana: int = VENTANA_ESTABILIDAD, votos_minimos: int = VOTOS_MINIMOS,
                 umbral: float = UMBRAL_REGISTRO, liberacion: int = INFERENCIAS_LIBERACION,
                 intervalo_minimo_s: float = INTERVALO_MIN_REGISTRO_S) -> None:
        self.ventana = ventana
        self.votos_minimos = votos_minimos
        self.umbral = umbral
        self.liberacion = liberacion
        self.intervalo_minimo_s = intervalo_minimo_s
        self.reiniciar()

    def reiniciar(self) -> None:
        self._historial: deque[tuple[str, float]] = deque(maxlen=self.ventana)
        self._bloqueado: str | None = None
        self._inestables = 0
        self._t_ultimo = -math.inf

    def actualizar(self, material: str, confianza: float,
                   ahora: float | None = None) -> tuple[float, tuple[str, float] | None]:
        """
        Returns:
            (estabilidad 0-1, evento) donde evento = (material, confianza_media)
            si hay que registrar, o None.
        """
        ahora = time.monotonic() if ahora is None else ahora
        self._historial.append((material, float(confianza)))
        lider, votos = Counter(m for m, _ in self._historial).most_common(1)[0]
        estabilidad = votos / self.ventana
        confianza_media = fmean(c for m, c in self._historial if m == lider)

        estable = (
            len(self._historial) == self.ventana
            and votos >= self.votos_minimos
            and lider not in self.NO_REGISTRABLES
            and confianza_media > self.umbral
        )
        if not estable:
            self._inestables += 1
            if self._inestables >= self.liberacion:
                self._bloqueado = None
            return estabilidad, None

        self._inestables = 0
        if lider == self._bloqueado or ahora - self._t_ultimo < self.intervalo_minimo_s:
            return estabilidad, None
        self._bloqueado = lider
        self._t_ultimo = ahora
        return estabilidad, (lider, round(confianza_media, 1))


class DetectorPresencia:
    """
    ¿Hay un objeto en la zona de interés?

    El clasificador MobileNetV2 es de conjunto cerrado: ante una mesa vacía
    igual responde "Vidrio 78 %". Sin esta compuerta el sistema registraría
    el fondo y nunca sabría que el objeto se retiró. Se usa sustracción de
    fondo MOG2 sobre la zona reducida a 160x120 (~1 ms por frame):
      * Calibración: los primeros frames aprenden el fondo vacío.
      * Sin objeto: aprende rápido (se adapta a cambios de luz).
      * Con objeto: aprende muy lento (no "absorbe" el residuo al fondo).
    """

    def __init__(self, frames_calibracion: int = FRAMES_CALIBRACION,
                 umbral: float = UMBRAL_PRESENCIA, histeresis: float = 0.015) -> None:
        self.frames_calibracion = frames_calibracion
        self.umbral = umbral
        self.histeresis = histeresis
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.reiniciar()

    def reiniciar(self) -> None:
        self._mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=36,
                                                       detectShadows=False)
        self._vistos = 0
        self.presente = False
        self.fraccion = 0.0

    @property
    def calibrando(self) -> bool:
        return self._vistos < self.frames_calibracion

    def actualizar(self, roi_bgr: np.ndarray) -> bool:
        gris = cv2.cvtColor(cv2.resize(roi_bgr, (160, 120), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (5, 5), 0)
        if self.calibrando:
            self._mog.apply(gris, learningRate=-1)
            self._vistos += 1
            self.presente, self.fraccion = False, 0.0
            return False
        mascara = self._mog.apply(gris, learningRate=0.0005 if self.presente else 0.02)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, self._kernel)
        self.fraccion = cv2.countNonZero(mascara) / mascara.size
        limite = self.umbral - self.histeresis if self.presente else self.umbral
        self.presente = self.fraccion > limite
        return self.presente


def recortar_roi(img: np.ndarray) -> np.ndarray:
    alto, ancho = img.shape[:2]
    x1, y1, x2, y2 = ROI
    return img[int(y1 * alto):int(y2 * alto), int(x1 * ancho):int(x2 * ancho)]


# ===========================================================================
# 4. PROCESADOR DE VIDEO WEBRTC
# ===========================================================================
@dataclass
class EstadoVivo:
    """Instantánea inmutable que el hilo de inferencia publica para la UI."""
    material: str = "—"
    confianza: float = 0.0
    contenedor: str = "—"
    motivo: str = ""
    forma: str = ""
    top_predicciones: list[tuple[str, float]] = field(default_factory=list)
    estabilidad: float = 0.0
    latencia_ms: float = 0.0
    fps_inferencia: float = 0.0
    fps_video: float = 0.0
    inferencias: int = 0
    registros: int = 0
    ultimo_registro: dict | None = None
    error: str | None = None
    calibrando: bool = True
    presente: bool = False
    fraccion_cambio: float = 0.0


def _ascii(texto: str) -> str:
    """cv2.putText solo dibuja ASCII: 'Cartón' -> 'Carton'."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


class ProcesadorReciclaje(VideoProcessorBase):
    """
    Procesador de video desacoplado:
      * recv() se ejecuta por cada frame en el hilo de aiortc y debe ser
        rapidísimo: guarda el frame más reciente y dibuja el último resultado.
      * La inferencia vive en un hilo daemon propio que siempre toma el frame
        más nuevo. Si la IA tarda 80 ms y la cámara entrega 30 fps, los frames
        intermedios se descartan en vez de acumularse (causa del "lag" creciente).
    """

    def __init__(self, motor_compartido: MotorCompartido, motor_inicial: str) -> None:
        self._ia = motor_compartido
        self._motor = motor_inicial
        self._estabilizador = EstabilizadorDetecciones()
        self._presencia = DetectorPresencia()
        self._pedir_recalibracion = False

        self._lock_frame = threading.Lock()
        self._frame_pendiente: np.ndarray | None = None
        self._hay_frame = threading.Event()

        self._lock_estado = threading.Lock()
        self._estado = EstadoVivo()
        self._t_frame_anterior: float | None = None
        self._fps_video = 0.0

        self._detener = threading.Event()
        self._hilo = threading.Thread(target=self._bucle_inferencia,
                                      name="inferencia-reciclaje", daemon=True)
        self._hilo.start()

    # ---- Interfaz con el hilo de Streamlit -------------------------------
    @property
    def motor(self) -> str:
        return self._motor

    @motor.setter
    def motor(self, valor: str) -> None:
        if valor != self._motor:
            self._motor = valor
            self._estabilizador.reiniciar()   # No mezclar votos de motores distintos

    def recalibrar(self) -> None:
        """Vuelve a aprender el fondo (llamar con la zona vacía)."""
        self._pedir_recalibracion = True

    def estado(self) -> EstadoVivo:
        with self._lock_estado:
            return replace(self._estado)

    # ---- Hilo de aiortc: un frame -----------------------------------------
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        imagen = frame.to_ndarray(format="bgr24")

        ahora = time.perf_counter()
        if self._t_frame_anterior is not None:
            dt = max(1e-3, ahora - self._t_frame_anterior)
            self._fps_video = 0.9 * self._fps_video + 0.1 * (1.0 / dt)
        self._t_frame_anterior = ahora

        # Copia reducida para la IA (cv2.resize devuelve un arreglo nuevo, así
        # el HUD que se dibuja abajo no contamina la entrada del modelo).
        alto, ancho = imagen.shape[:2]
        if ancho > ANCHO_INFERENCIA:
            escala = ANCHO_INFERENCIA / ancho
            reducida = cv2.resize(imagen, (ANCHO_INFERENCIA, int(alto * escala)),
                                  interpolation=cv2.INTER_AREA)
        else:
            reducida = imagen.copy()
        with self._lock_frame:
            self._frame_pendiente = reducida      # Sobrescribe: buzón de 1 frame
        self._hay_frame.set()

        self._dibujar_hud(imagen, self.estado())
        return av.VideoFrame.from_ndarray(imagen, format="bgr24")

    def on_ended(self) -> None:
        self._detener.set()
        self._hay_frame.set()

    # ---- Hilo de inferencia -------------------------------------------------
    def _bucle_inferencia(self) -> None:
        while not self._detener.is_set():
            if not self._hay_frame.wait(timeout=0.5):
                continue
            self._hay_frame.clear()
            with self._lock_frame:
                imagen, self._frame_pendiente = self._frame_pendiente, None
            if imagen is None:
                continue

            t0 = time.perf_counter()
            if self._pedir_recalibracion:
                self._pedir_recalibracion = False
                self._presencia.reiniciar()
                self._estabilizador.reiniciar()
            roi = recortar_roi(imagen)
            presente = self._presencia.actualizar(roi)
            if not presente:
                # Zona vacía o calibrando: no se gasta CPU en la red neuronal.
                self._estabilizador.escena_vacia()
                with self._lock_estado:
                    e = self._estado
                    e.calibrando, e.presente = self._presencia.calibrando, False
                    e.fraccion_cambio = self._presencia.fraccion
                    e.estabilidad, e.fps_video = 0.0, self._fps_video
                self._detener.wait(INTERVALO_INFERENCIA_S)
                continue

            motor = self._motor
            try:
                pred = self._ia.predecir(roi, motor)
            except Exception as exc:  # El hilo nunca debe morir en silencio
                with self._lock_estado:
                    self._estado.error = f"Inferencia: {exc}"
                self._detener.wait(0.5)
                continue
            latencia = time.perf_counter() - t0

            estabilidad, evento = self._estabilizador.actualizar(pred.material, pred.confianza)
            registro, error = None, None
            if evento is not None:
                material, confianza = evento
                contenedor = CATEGORIAS[material]["contenedor"]
                try:
                    id_reg = db.crear_clasificacion(material, confianza, contenedor,
                                                    origen=origen_registro(motor), motor=motor)
                    registro = {"id": id_reg, "material": material, "confianza": confianza,
                                "contenedor": contenedor, "hora": datetime.now().strftime("%H:%M:%S"),
                                "t": time.monotonic()}
                except Exception as exc:
                    error = f"SQLite: {exc}"

            with self._lock_estado:
                e = self._estado
                e.material, e.confianza, e.contenedor = pred.material, pred.confianza, pred.contenedor
                e.motivo, e.forma = pred.motivo, pred.forma
                e.top_predicciones = list(pred.top_predicciones)
                e.estabilidad = estabilidad
                e.latencia_ms = 0.8 * e.latencia_ms + 0.2 * latencia * 1000 if e.inferencias else latencia * 1000
                ciclo = max(latencia, INTERVALO_INFERENCIA_S)
                e.fps_inferencia = 0.8 * e.fps_inferencia + 0.2 * (1.0 / ciclo) if e.inferencias else 1.0 / ciclo
                e.fps_video = self._fps_video
                e.inferencias += 1
                e.error = error
                e.calibrando, e.presente = False, True
                e.fraccion_cambio = self._presencia.fraccion
                if registro:
                    e.registros += 1
                    e.ultimo_registro = registro

            # Limita la tasa de inferencia para dejar CPU al códec de video.
            resto = INTERVALO_INFERENCIA_S - (time.perf_counter() - t0)
            if resto > 0:
                self._detener.wait(resto)

    # ---- HUD (barato: solo dibuja sobre una franja) -------------------------
    @staticmethod
    def _dibujar_hud(img: np.ndarray, e: EstadoVivo) -> None:
        alto, ancho = img.shape[:2]
        x1, y1, x2, y2 = ROI
        color_roi = (90, 200, 30) if e.presente else (235, 235, 235)
        cv2.rectangle(img, (int(x1 * ancho), int(y1 * alto)), (int(x2 * ancho), int(y2 * alto)),
                      color_roi, 1, cv2.LINE_AA)

        franja = img[0:40, :]
        franja[:] = (franja * 0.35).astype(np.uint8)          # fondo oscuro translúcido
        if e.calibrando:
            texto = "Calibrando fondo: deja el recuadro vacio..."
        elif not e.presente:
            texto = "Coloca el residuo dentro del recuadro"
        else:
            texto = f"{_ascii(e.material)}  {e.confianza:.0f}%  -> {_ascii(e.contenedor)}"
        cv2.putText(img, texto, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Barra de estabilidad inferior
        largo = int(ancho * e.estabilidad)
        listo = e.estabilidad >= VOTOS_MINIMOS / VENTANA_ESTABILIDAD and e.confianza > UMBRAL_REGISTRO
        cv2.rectangle(img, (0, alto - 6), (largo, alto), (90, 200, 30) if listo else (0, 165, 255), -1)

        # Marco verde durante 1.5 s tras guardar un registro
        reg = e.ultimo_registro
        if reg and time.monotonic() - reg["t"] < 1.5:
            cv2.rectangle(img, (2, 42), (ancho - 3, alto - 8), (90, 200, 30), 4)
            cv2.putText(img, "REGISTRADO", (ancho - 175, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (90, 200, 30), 2, cv2.LINE_AA)


# ===========================================================================
# 5. COMPONENTES VISUALES
# ===========================================================================
def tarjeta(etiqueta: str, valor_html: str, subtitulo: str = "") -> None:
    sub = f'<div class="card-sub">{subtitulo}</div>' if subtitulo else ""
    st.markdown(
        f'<div class="card"><div class="card-label">{etiqueta}</div>'
        f'<div class="card-value">{valor_html}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def badge_contenedor(contenedor: str) -> str:
    color = COLORES_CONTENEDOR.get(contenedor, "#E5E7EB")
    return (f'<span class="contenedor-badge"><span class="contenedor-dot" '
            f'style="background:{color};"></span>Contenedor {contenedor}</span>')


def barra_confianza(confianza: float) -> str:
    ancho = max(0.0, min(100.0, confianza))
    return (f'{confianza:.1f}%<div class="barra-fondo">'
            f'<div class="barra-relleno" style="width:{ancho}%;"></div></div>')


def tarjetas_resultado(material: str, confianza: float, contenedor: str,
                       subtitulo_conf: str = "") -> None:
    descripcion = CATEGORIAS.get(material, {}).get("descripcion", "")
    tarjeta("Material detectado", material, descripcion)
    tarjeta("Contenedor asignado (Res. 2184 de 2019)", badge_contenedor(contenedor),
            "Blanco: aprovechables · Negro: no aprovechables · Verde: orgánicos")
    tarjeta("Confianza del modelo", barra_confianza(confianza), subtitulo_conf)


def panel_diagnostico(top3: list[tuple[str, float]], motivo: str, forma: str) -> None:
    with st.container(border=True):
        st.markdown("**🛠️ Diagnóstico técnico**")
        st.caption(f"Forma: {forma or 'N/A'} · {motivo}")
        if top3:
            cols = st.columns(len(top3))
            for col, (clase, pct) in zip(cols, top3):
                col.metric(clase, f"{pct:.1f}%")
        else:
            st.caption("El motor actual no usa la red neuronal (sin Top 3).")


def tabla_registros(df: pd.DataFrame) -> None:
    st.dataframe(
        df[["id", "fecha_hora", "material", "confianza", "contenedor", "origen", "motor"]],
        hide_index=True,
        width="stretch",
        column_config={
            "id": "ID",
            "fecha_hora": st.column_config.DatetimeColumn("Fecha y hora", format="DD/MM/YYYY HH:mm:ss"),
            "material": "Material",
            "confianza": st.column_config.ProgressColumn("Confianza", min_value=0, max_value=100,
                                                         format="%.1f%%"),
            "contenedor": "Contenedor",
            "origen": "Origen",
            "motor": "Motor",
        },
    )


# ===========================================================================
# 6. PANTALLA 1 — CLASIFICADOR EN VIVO
# ===========================================================================
def _panel_en_vivo(ctx, diagnostico: bool) -> None:
    """Se re-ejecuta sola cada segundo (st.fragment) sin tocar el resto de la página."""
    proc: ProcesadorReciclaje | None = ctx.video_processor
    if not ctx.state.playing or proc is None:
        tarjeta("Esperando video",
                "Pulsa <b>START</b>",
                "Ubica el residuo frente a la cámara. Se registra automáticamente cuando "
                f"la detección es estable y supera el {UMBRAL_REGISTRO:.0f}% de confianza.")
        return

    e = proc.estado()
    if e.error:
        st.error(e.error)
    reg = e.ultimo_registro
    if reg and time.monotonic() - reg["t"] < 4:
        st.success(f"💾 Registro #{reg['id']} guardado: **{reg['material']}** "
                   f"({reg['confianza']:.1f}%) → contenedor **{reg['contenedor']}**")
    if e.calibrando:
        st.info("⏳ Calibrando el fondo: mantén el recuadro **vacío** un momento.")
        return
    if not e.presente:
        tarjeta("Zona vacía", "Coloca un residuo en el recuadro",
                f"Cambio detectado en la zona: {e.fraccion_cambio:.1%} "
                f"(umbral {UMBRAL_PRESENCIA:.0%}) · Registros de esta sesión: {e.registros}")
        return

    tarjetas_resultado(e.material, e.confianza, e.contenedor,
                       f"<b>Motor:</b> {proc.motor} · <b>Forma:</b> {e.forma}")

    st.progress(e.estabilidad, text=f"Estabilidad de la detección: {e.estabilidad:.0%} "
                                    f"(se requiere {VOTOS_MINIMOS}/{VENTANA_ESTABILIDAD})")
    st.caption(f"🎞️ Video **{e.fps_video:.0f} fps** · 🧠 IA **{e.fps_inferencia:.1f} inf/s** · "
               f"⏱️ Latencia **{e.latencia_ms:.0f} ms** · 💾 Registros de la sesión **{e.registros}**")

    if diagnostico:
        panel_diagnostico(e.top_predicciones, e.motivo, e.forma)


def modo_video(motor_ia: MotorCompartido, motor: str, diagnostico: bool) -> None:
    col_video, col_panel = st.columns([1.15, 1], gap="large")
    with col_video:
        ctx = webrtc_streamer(
            key="clasificador-reciclaje",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURACION,
            media_stream_constraints=RESTRICCIONES_MEDIA,
            video_processor_factory=lambda: ProcesadorReciclaje(motor_ia, motor),
            async_processing=True,   # recv() fuera del event loop; frames lentos se descartan
        )
        # Propaga el motor elegido en la barra lateral al procesador en caliente.
        if ctx.video_processor is not None:
            ctx.video_processor.motor = motor
            if st.button("🔄 Recalibrar fondo", help="Úsalo con el recuadro vacío si cambió la "
                                                     "iluminación o se movió la cámara."):
                ctx.video_processor.recalibrar()
        st.caption("🔒 La cámara solo se activa al pulsar START. El video no se guarda: "
                   "solo se registra el material detectado.")

    with col_panel:
        st.fragment(_panel_en_vivo, run_every=1.0 if ctx.state.playing else None)(ctx, diagnostico)

    st.markdown("#### 🕘 Últimos registros de la cámara")
    st.fragment(_ultimos_camara, run_every=3.0 if ctx.state.playing else None)()


def _ultimos_camara() -> None:
    df = db.listar_clasificaciones(10, origenes=["camara"])
    if df.empty:
        st.caption("Todavía no hay registros de la cámara.")
    else:
        tabla_registros(df)


def modo_foto(motor_ia: MotorCompartido, motor: str, diagnostico: bool) -> None:
    """Respaldo para redes donde WebRTC no conecta (sin HTTPS, NAT estricto)."""
    col_cam, col_res = st.columns(2, gap="large")
    with col_cam:
        foto = st.camera_input("Toma una foto del residuo")
    with col_res:
        if foto is None:
            tarjeta("Esperando captura", "📸", "Toma una foto para clasificarla.")
            return
        huella = hashlib.md5(foto.getvalue(), usedforsecurity=False).hexdigest()
        clave = (huella, motor)
        if st.session_state.get("foto_clave") != clave:
            pred = motor_ia.predecir(foto, motor)
            id_reg = None
            if pred.confianza > UMBRAL_REGISTRO and pred.material != "No identificado" \
                    and st.session_state.get("foto_huella_guardada") != huella:
                id_reg = db.crear_clasificacion(pred.material, pred.confianza, pred.contenedor,
                                                origen=origen_registro(motor), motor=motor)
                st.session_state.foto_huella_guardada = huella
            st.session_state.foto_clave = clave
            st.session_state.foto_pred = (pred, id_reg)
        pred, id_reg = st.session_state.foto_pred
        if id_reg:
            st.success(f"💾 Registro #{id_reg} guardado en la base de datos.")
        else:
            st.warning(f"No se registró: confianza ≤ {UMBRAL_REGISTRO:.0f}% o foto ya registrada.")
        tarjetas_resultado(pred.material, pred.confianza, pred.contenedor, pred.motivo)
        if pred.imagen_anotada is not None:
            st.image(pred.imagen_anotada, width="stretch")
        if diagnostico:
            panel_diagnostico(pred.top_predicciones, pred.motivo, pred.forma)


def pantalla_clasificador(motor_ia: MotorCompartido, motor: str, diagnostico: bool) -> None:
    st.markdown('<div class="titulo-app">📷 Clasificador en Vivo</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitulo-app">Motor activo: <b>{motor}</b>. '
                f'Se guarda en la base de datos cada residuo detectado de forma estable '
                f'con confianza superior al {UMBRAL_REGISTRO:.0f}%.</p>', unsafe_allow_html=True)
    modo = st.segmented_control("Modo de captura", [MODO_VIDEO, MODO_FOTO],
                                default=MODO_VIDEO, label_visibility="collapsed")
    if modo == MODO_FOTO:
        modo_foto(motor_ia, motor, diagnostico)
    else:
        modo_video(motor_ia, motor, diagnostico)


# ===========================================================================
# 7. PANTALLA 2 — TABLERO ESTADÍSTICO
# ===========================================================================
@st.cache_data(ttl=600, show_spinner=False, max_entries=32)
def cargar_datos_tablero(firma: tuple[int, int], origenes: tuple[str, ...]) -> dict:
    """
    Agregaciones en SQL, cacheadas por la `firma` de la tabla: mientras nadie
    escriba, el tablero no vuelve a consultar SQLite.
    """
    return {
        "kpis": db.resumen_kpis(origenes),
        "tipos": db.proporcion_aprovechable(origenes),
        "materiales": db.conteo_por_material(origenes),
        "dias": db.volumen_por_dia(origenes),
        "validacion": db.datos_validacion(origenes),
        "ultimos": db.listar_clasificaciones(15, origenes=origenes),
    }


def _estilo_base(fig: go.Figure, alto: int = 360) -> go.Figure:
    fig.update_layout(
        height=alto,
        margin=dict(t=56, b=8, l=8, r=8),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="white", font_size=13),
        title=dict(x=0, xanchor="left", font_size=16),
    )
    return fig


def grafico_dona(df_tipos: pd.DataFrame) -> go.Figure:
    df = df_tipos.set_index("tipo").reindex([t for t in ORDEN_TIPOS if t in set(df_tipos["tipo"])]).reset_index()
    total = int(df["cantidad"].sum())
    pct_aprov = 100 * df.loc[df["tipo"] == "Aprovechable", "cantidad"].sum() / max(1, total)
    fig = go.Figure(go.Pie(
        labels=df["tipo"], values=df["cantidad"], hole=0.62, sort=False, direction="clockwise",
        marker=dict(colors=[COLORES_TIPO[t] for t in df["tipo"]], line=dict(color="white", width=2)),
        textinfo="percent", textposition="outside",
        hovertemplate="<b>%{label}</b><br>%{value:,} residuos<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{pct_aprov:.1f}%</b><br><span style='font-size:12px'>aprovechable</span>",
                       showarrow=False, font=dict(size=26))
    fig.update_layout(title_text="Reciclables vs. no aprovechables")
    _estilo_base(fig, alto=380)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
                      margin=dict(t=56, b=40, l=8, r=8))
    return fig


def grafico_materiales(df_mat: pd.DataFrame) -> go.Figure:
    df = df_mat.groupby(["material", "tipo"], as_index=False)["cantidad"].sum()
    df = df.sort_values("cantidad")
    fig = px.bar(df, x="cantidad", y="material", color="tipo", orientation="h",
                 color_discrete_map=COLORES_TIPO, category_orders={"tipo": ORDEN_TIPOS},
                 text="cantidad", title="Volumen detectado por material")
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="<b>%{y}</b><br>%{x:,} residuos<extra></extra>")
    fig.update_layout(barcornerradius=4, bargap=0.35, xaxis_title=None, yaxis_title=None,
                      yaxis=dict(categoryorder="total ascending"))
    fig.update_xaxes(gridcolor=COLOR_GRILLA, zeroline=False)
    return _estilo_base(fig, alto=max(300, 60 + 42 * df["material"].nunique()))


def grafico_dias(df_dias: pd.DataFrame) -> go.Figure:
    fig = px.bar(df_dias, x="dia", y="cantidad", color="tipo", color_discrete_map=COLORES_TIPO,
                 category_orders={"tipo": ORDEN_TIPOS}, title="Volumen detectado por día")
    fig.update_traces(marker_line_color="white", marker_line_width=1,
                      hovertemplate="%{x|%d/%m/%Y}<br>%{y:,} residuos<extra></extra>")
    fig.update_layout(barmode="stack", barcornerradius=4, bargap=0.25,
                      xaxis_title=None, yaxis_title="Residuos", hovermode="x unified")
    fig.update_yaxes(gridcolor=COLOR_GRILLA, zeroline=False)
    fig.update_xaxes(tickformat="%d %b")
    return _estilo_base(fig, alto=320)


def seccion_validacion(df_val: pd.DataFrame) -> None:
    """Exactitud del modelo frente a la etiqueta real de TrashNet/TACO."""
    df = df_val.copy()
    df["predicho"] = df["material"].map(db.grupo_material)
    # La etiqueta real puede ser una clase fina («No aprovechable (Multicapa)»): se compara
    # por grupo, pero la matriz conserva la clase fina para exponer la brecha de dominio.
    df["acierto"] = df["predicho"] == df["etiqueta_real"].map(db.grupo_material)

    resumen = (df.assign(aciertos=df["cantidad"] * df["acierto"])
                 .groupby(["origen", "motor"], as_index=False)[["cantidad", "aciertos"]].sum())
    resumen["exactitud"] = 100 * resumen["aciertos"] / resumen["cantidad"]

    st.markdown("#### 🧪 Validación con bancos públicos")
    st.caption("Solo incluye registros con etiqueta real (ingestados con `ingesta_publica.py`). "
               "PET y PEAD se agrupan como «Plástico». Si se ingestó con umbral, la exactitud "
               "es sobre las detecciones que lo superaron.")
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.dataframe(resumen.drop(columns="aciertos"), hide_index=True, width="stretch", column_config={
            "origen": "Dataset", "motor": "Motor", "cantidad": "Muestras",
            "exactitud": st.column_config.ProgressColumn("Exactitud", min_value=0, max_value=100,
                                                         format="%.1f%%"),
        })
    with c2:
        matriz = df.pivot_table(index="etiqueta_real", columns="predicho", values="cantidad",
                                aggfunc="sum", fill_value=0)
        fig = px.imshow(matriz, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                        labels=dict(x="Predicho", y="Real", color="Residuos"),
                        title="Matriz de confusión")
        fig.update_traces(hovertemplate="Real: %{y}<br>Predicho: %{x}<br>%{z:,}<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_estilo_base(fig, alto=380), width="stretch")


def _cuerpo_tablero(origenes: tuple[str, ...]) -> None:
    datos = cargar_datos_tablero(db.firma_datos(), origenes)
    k = datos["kpis"]
    if not k["total"]:
        st.info("No hay clasificaciones para el filtro seleccionado. Usa el **Clasificador en Vivo** "
                "o ejecuta `python ingesta_publica.py` para poblar la base de datos.")
        return

    mat = datos["materiales"]
    top = mat.groupby("material")["cantidad"].sum().sort_values(ascending=False)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("♻️ Total de residuos", f"{k['total']:,}".replace(",", "."))
    m2.metric("🎯 Confianza promedio", f"{k['confianza_promedio']:.1f}%")
    m3.metric("🔵 Aprovechables", f"{100 * (k['aprovechables'] or 0) / k['total']:.1f}%")
    m4.metric("🏆 Material más frecuente", top.index[0], f"{int(top.iloc[0]):,} unidades".replace(",", "."),
              delta_color="off", delta_arrow="off")

    g1, g2 = st.columns([1, 1.25], gap="large")
    with g1, st.container(border=True):
        st.plotly_chart(grafico_dona(datos["tipos"]), width="stretch")
    with g2, st.container(border=True):
        st.plotly_chart(grafico_materiales(mat), width="stretch")

    if datos["dias"]["dia"].nunique() > 1:
        with st.container(border=True):
            st.plotly_chart(grafico_dias(datos["dias"]), width="stretch")

    if not datos["validacion"].empty:
        seccion_validacion(datos["validacion"])

    st.markdown("#### 📋 Últimas 15 clasificaciones")
    tabla_registros(datos["ultimos"])


def pantalla_tablero() -> None:
    st.markdown('<div class="titulo-app">📊 Tablero Estadístico</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitulo-app">Indicadores alimentados por la base de datos SQLite.</p>',
                unsafe_allow_html=True)

    disponibles = db.origenes_disponibles()
    f1, f2, f3 = st.columns([3, 1.2, 1], vertical_alignment="bottom")
    origenes = f1.multiselect("Origen de los datos", disponibles, default=disponibles,
                              help="'camara' = clasificaciones reales; 'trashnet'/'taco' = ingesta "
                                   "de bancos públicos.")
    auto = f2.toggle("Auto-actualizar (5 s)", value=False)
    if origenes:
        csv = db.listar_clasificaciones(origenes=origenes).to_csv(index=False).encode("utf-8-sig")
        f3.download_button("⬇️ CSV", csv, "clasificaciones.csv", "text/csv", width="stretch")

    st.fragment(_cuerpo_tablero, run_every=5.0 if auto else None)(tuple(origenes))


# ===========================================================================
# 8. BARRA LATERAL Y PUNTO DE ENTRADA
# ===========================================================================
def _contador_bd() -> None:
    total, _ = db.firma_datos()
    st.markdown(f"**Registros en BD:** {total:,}".replace(",", "."))


def barra_lateral(motor_ia: MotorCompartido) -> tuple[str, str, bool]:
    with st.sidebar:
        st.markdown("## ♻️ Reciclaje Inteligente")
        st.caption("Clasificación de residuos y analítica en tiempo real")
        pantalla = st.radio("Navegación", [PANTALLA_CLASIFICADOR, PANTALLA_TABLERO])

        st.divider()
        st.markdown("### ⚙️ Motor de Clasificación")
        motor = st.radio("Seleccionar estrategia:", MOTORES, index=0,
                         help="Se puede cambiar en caliente mientras el video está activo.")
        if MODO_DIDACTICO:
            st.info("🎓 Modo didáctico activo: Color y Geometría reproducen fallos conocidos. "
                    f"Sus eventos se guardan con origen '{ORIGEN_DIDACTICO}'.")
            if motor in MOTORES_DIDACTICOS:
                st.warning("Este motor NO es fiable para separar residuos reales.")
        diagnostico = st.checkbox("🛠️ Activar Modo Diagnóstico", value=False)

        st.divider()
        st.markdown(f"**Sistema:** {motor_ia.nombre}")
        st.fragment(_contador_bd, run_every=5.0)()
        if motor_ia.clasificador.red_onnx is None:
            st.warning(f"Red neuronal no disponible: {motor_ia.clasificador.error_carga_dnn}")

        with st.popover("🗑️ Gestionar datos", width="stretch"):
            origenes = db.origenes_disponibles()
            if origenes:
                origen = st.selectbox("Borrar registros de un origen", origenes)
                if st.button(f"Borrar '{origen}'", width="stretch"):
                    st.toast(f"Eliminados {db.eliminar_por_origen(origen)} registros de '{origen}'.")
                    st.rerun()
            if st.button("⚠️ Vaciar toda la base de datos", type="primary", width="stretch"):
                st.toast(f"Eliminados {db.reiniciar()} registros.")
                st.rerun()
        st.caption("Desarrollado por Johan Lasso")
    return pantalla, motor, diagnostico


def main() -> None:
    cargar_css(RUTA_CSS)
    motor_ia = cargar_motor()
    pantalla, motor, diagnostico = barra_lateral(motor_ia)
    if pantalla == PANTALLA_CLASIFICADOR:
        pantalla_clasificador(motor_ia, motor, diagnostico)
    else:
        pantalla_tablero()


if __name__ == "__main__":
    main()
