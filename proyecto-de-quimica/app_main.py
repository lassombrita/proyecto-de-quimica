"""
app_main.py — Interfaz central del Sistema de Reciclaje Inteligente (Pipeline Edge)
==================================================================================
Ejecución:   streamlit run app_main.py

Pantallas (navegación lateral):
    1. "📡 Escáner en Vivo"                 Video WebRTC asíncrono + bounding boxes.
    2. "📊 Central Analítica (Dashboard)"    KPIs y gráficos Plotly por color de la
                                             Resolución 2184 de 2019 (Blanco/Negro/Verde).

Modelo de concurrencia de la interfaz:
    * El script de Streamlit se re-ejecuta en el hilo de la sesión. Lo pesado y
      compartido (modelo de visión, hilo escritor de SQLite) se crea UNA vez por
      proceso con @st.cache_resource y se reutiliza en todas las sesiones/reruns.
    * El video NO provoca reruns: los frames viven en el hilo de streamlit-webrtc.
      La UI consulta métricas con `st.fragment(run_every=...)`, que re-renderiza
      SOLO el panel afectado (no el componente de video ni el resto de la página).
    * El tablero lee con `database.get_stats()`: una transacción de lectura WAL que
      nunca bloquea al hilo escritor alimentado por la IA.

Autor: Johan Lasso
"""

from __future__ import annotations

import html
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import database as db
from vision_edge import (
    CERTIDUMBRE_MAX,
    CERTIDUMBRE_MIN,
    TAXONOMIA,
    ModeloVisionSimulado,
    ProcesadorResiduosEdge,
)

# ===========================================================================
# 1. CONFIGURACIÓN GENERAL
# ===========================================================================
st.set_page_config(
    page_title="Reciclaje Inteligente · Edge",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PANTALLA_ESCANER = "📡 Escáner en Vivo"
PANTALLA_DASHBOARD = "📊 Central Analítica (Dashboard)"

# Colores de la Resolución 2184 de 2019 para los gráficos. El "blanco" se pinta
# gris muy claro con borde oscuro: un blanco puro desaparecería sobre el fondo.
COLOR_HEX = {"BLANCO": "#E9EDF2", "NEGRO": "#1F2937", "VERDE": "#16A34A"}
ETIQUETA_COLOR = {
    "BLANCO": "Blanco · Aprovechable",
    "NEGRO": "Negro · No aprovechable",
    "VERDE": "Verde · Orgánico",
}
COLOR_POR_ETIQUETA = {ETIQUETA_COLOR[k]: v for k, v in COLOR_HEX.items()}
ORDEN_ETIQUETAS = list(ETIQUETA_COLOR.values())
BORDE_MARCAS = "#475569"

# STUN público: necesario si el navegador no corre en la misma máquina (Jetson en
# la red local). Detrás de NAT simétrico (4G, redes universitarias) hace falta TURN.
RTC_CONFIGURACION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# 640x480 basta: la IA reduce a 320 px y aiortc codifica VP8 por software en la CPU.
RESTRICCIONES_MEDIA = {
    "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
              "frameRate": {"ideal": 24, "max": 30}},
    "audio": False,
}

# ===========================================================================
# 2. ESTILOS (CSS nativo inyectado)
# ===========================================================================
CSS = """
<style>
/* --- Ocultar el cromo predeterminado de Streamlit -------------------------
   Se ocultan menú, barra de herramientas, decoración, botón Deploy y footer.
   El <header> se deja TRANSPARENTE en lugar de display:none porque en las
   versiones recientes contiene el botón para re-expandir la barra lateral; si
   se ocultara, un usuario que cierre el sidebar no podría volver a abrirlo. */
#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; height: 2.5rem; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }

/* --- Tipografía ----------------------------------------------------------- */
.titulo-app  { font-size: 1.75rem; font-weight: 750; color: #0F172A; letter-spacing: -0.02em; margin: 0; }
.subtitulo-app { color: #64748B; margin: 0.15rem 0 1.1rem 0; font-size: 0.95rem; }

/* --- Tarjetas profesionales ---------------------------------------------- */
.tarjeta {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
    padding: 1rem 1.15rem; margin-bottom: 0.85rem;
    box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 14px rgba(15,23,42,.06);
}
.tarjeta-etiqueta { font-size: .74rem; text-transform: uppercase; letter-spacing: .06em;
                    color: #64748B; font-weight: 650; margin-bottom: .3rem; }
.tarjeta-valor { font-size: 1.55rem; font-weight: 750; color: #0F172A; line-height: 1.2; }
.tarjeta-sub   { font-size: .82rem; color: #64748B; margin-top: .35rem; }

/* Contenedores con key="tarjeta_*": envuelven gráficos/tablas como tarjetas. */
[class*="st-key-tarjeta"] {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
    padding: .9rem 1rem .4rem 1rem;
    box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 14px rgba(15,23,42,.06);
}

/* Chip del contenedor (Res. 2184) */
.chip { display: inline-flex; align-items: center; gap: .5rem; padding: .3rem .8rem;
        border-radius: 999px; font-weight: 650; font-size: .95rem; border: 1px solid #CBD5E1; }
.chip-dot { width: .95rem; height: .95rem; border-radius: 50%; border: 1px solid #475569; }

/* Barra de certidumbre */
.barra { height: 8px; border-radius: 99px; background: #E2E8F0; margin-top: .5rem; overflow: hidden; }
.barra > div { height: 100%; border-radius: 99px; background: #1B7F5A; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
</style>
"""


# ===========================================================================
# 3. RECURSOS COMPARTIDOS (@st.cache_resource: una vez por proceso)
# ===========================================================================
@st.cache_resource(show_spinner="🧠 Cargando modelo de visión...")
def cargar_modelo() -> ModeloVisionSimulado:
    """
    El modelo se instancia una sola vez y se comparte entre sesiones. Con un
    modelo real (TensorRT/ONNX) esto evita recargar pesos en cada rerun, que en
    una Jetson puede costar varios segundos y cientos de MB de memoria unificada.
    """
    return ModeloVisionSimulado()


@st.cache_resource(show_spinner=False)
def cargar_registrador() -> db.RegistradorAsincrono:
    """Inicializa la BD (WAL) y arranca el hilo escritor único del proceso."""
    db.init_db()
    return db.obtener_registrador()


@st.cache_resource(show_spinner=False)
def registrar_plantilla_plotly() -> str:
    """Registra una sola vez la plantilla visual común a todos los gráficos."""
    base = pio.templates["plotly_white"]
    plantilla = base.to_plotly_json()
    plantilla["layout"].update(
        font={"family": "Inter, Segoe UI, sans-serif", "size": 13, "color": "#1F2937"},
        margin={"l": 10, "r": 10, "t": 46, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0,
                "title": {"text": ""}},
        hoverlabel={"bgcolor": "#FFFFFF", "font": {"color": "#0F172A"}},
    )
    pio.templates["reciclaje"] = plantilla
    return "reciclaje"


# ===========================================================================
# 4. COMPONENTES VISUALES
# ===========================================================================
def tarjeta(etiqueta: str, valor_html: str, sub: str = "") -> None:
    sub_html = f'<div class="tarjeta-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="tarjeta"><div class="tarjeta-etiqueta">{etiqueta}</div>'
                f'<div class="tarjeta-valor">{valor_html}</div>{sub_html}</div>',
                unsafe_allow_html=True)


def chip_contenedor(contenedor: str) -> str:
    color = db.color_de_contenedor(contenedor)
    return (f'<span class="chip"><span class="chip-dot" style="background:{COLOR_HEX[color]}">'
            f'</span>{html.escape(contenedor)}</span>')


def barra(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    return f'{pct:.1f}%<div class="barra"><div style="width:{pct}%"></div></div>'


def encabezado(titulo: str, subtitulo: str) -> None:
    st.markdown(f'<p class="titulo-app">{titulo}</p><p class="subtitulo-app">{subtitulo}</p>',
                unsafe_allow_html=True)


# ===========================================================================
# 5. PANTALLA 1 — ESCÁNER EN VIVO
# ===========================================================================
def _panel_en_vivo(ctx, registrador: db.RegistradorAsincrono) -> None:
    """Fragmento que se refresca solo (1 s) sin re-ejecutar el componente de video."""
    proc: ProcesadorResiduosEdge | None = ctx.video_processor
    if not ctx.state.playing or proc is None:
        tarjeta("Cámara detenida", "Pulsa <b>START</b>",
                "El análisis corre en un hilo asíncrono: la interfaz no se recarga por fotograma.")
        return

    m = proc.metricas()
    if m.error:
        st.error(m.error)
    d = m.ultima
    if d is None:
        tarjeta("Buscando objeto", "Acerca un residuo a la cámara",
                "Se necesita un contorno que ocupe al menos el 2 % del encuadre.")
    else:
        tarjeta("Objeto detectado", html.escape(d.material))
        tarjeta("Contenedor destino · Res. 2184 de 2019", chip_contenedor(d.contenedor))
        tarjeta("Certidumbre del modelo", barra(d.certidumbre))

    c1, c2, c3 = st.columns(3)
    c1.metric("Video", f"{m.fps_video:.0f} fps")
    c2.metric("Inferencia", f"{m.fps_inferencia:.1f} inf/s")
    c3.metric("Latencia IA", f"{m.latencia_ms:.0f} ms")
    r = registrador.metricas()
    st.caption(f"🎞️ Frames descartados por el búfer (anti buffer-bloat): **{m.frames_descartados}** de "
               f"{m.frames_recibidos} · 💾 Eventos encolados en esta sesión: **{m.eventos_encolados}** · "
               f"Escritos (proceso): **{r['escritos']}** · Pendientes: **{r['pendientes']}**")


def pantalla_escaner(modelo: ModeloVisionSimulado, registrador: db.RegistradorAsincrono,
                     eventos_por_s: float, umbral: float, hud: bool) -> None:
    encabezado("📡 Escáner en Vivo",
               "Flujo WebRTC asíncrono con inferencia en hilo dedicado y registro no bloqueante en SQLite (WAL).")
    col_video, col_panel = st.columns([1.35, 1], gap="large")
    intervalo = 1.0 / eventos_por_s

    with col_video:
        with st.container(key="tarjeta_video"):
            ctx = webrtc_streamer(
                key="escaner-edge",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURACION,
                media_stream_constraints=RESTRICCIONES_MEDIA,
                video_processor_factory=lambda: ProcesadorResiduosEdge(
                    modelo, registrador, intervalo_registro_s=intervalo,
                    umbral_registro=umbral, mostrar_hud=hud),
                # recv() corre fuera del event loop de aiortc; si se atrasa, streamlit-webrtc
                # entrega solo el frame más reciente (segunda capa anti-latencia).
                async_processing=True,
            )
            # Parámetros en caliente: asignaciones atómicas leídas por los otros hilos.
            proc = ctx.video_processor
            if proc is not None:
                proc.intervalo_registro_s = intervalo
                proc.umbral_registro = umbral
                proc.mostrar_hud = hud
                proc.pausado = st.toggle("⏸️ Pausar inferencia", value=False,
                                         help="El video sigue, pero la IA deja de analizar y registrar.")
            st.caption("🔒 El video no se almacena: solo se registra el material, el contenedor y la certidumbre.")

    with col_panel:
        st.fragment(_panel_en_vivo, run_every=1.0 if ctx.state.playing else None)(ctx, registrador)


# ===========================================================================
# 6. PANTALLA 2 — CENTRAL ANALÍTICA (DASHBOARD)
# ===========================================================================
def _aplicar_etiquetas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Contenedor"] = df["color"].map(ETIQUETA_COLOR).fillna(df["color"])
    return df


def _estilo(fig, titulo: str, alto: int = 360):
    fig.update_layout(template=registrar_plantilla_plotly(), title={"text": titulo, "x": 0, "font": {"size": 15}},
                      height=alto)
    return fig


def _tablero(granularidad: str, ventana: str) -> None:
    """Fragmento del tablero: se re-ejecuta solo si el auto-refresco está activo."""
    desde = {"Todo": None, "Última hora": timedelta(hours=1),
             "Últimas 24 h": timedelta(days=1), "Últimos 7 días": timedelta(days=7)}[ventana]
    try:
        stats = db.get_stats(granularidad, desde=datetime.now() - desde if desde else None)
    except Exception as exc:   # p. ej. disco lleno: se informa sin tumbar la app
        st.error(f"No se pudo leer la base de datos: {exc}")
        return

    total = int(stats["total"] or 0)
    if total == 0:
        tarjeta("Sin datos", "Aún no hay clasificaciones",
                "Inicia el Escáner en Vivo para que el hilo de IA empiece a registrar eventos.")
        return

    por_cont = _aplicar_etiquetas(stats["por_contenedor"])
    por_color = por_cont.groupby("Contenedor", as_index=False)["cantidad"].sum()
    blancos = int(por_cont.loc[por_cont["color"] == "BLANCO", "cantidad"].sum())

    # --- KPIs -----------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        tarjeta("Eventos registrados", f"{total:,}".replace(",", "."))
    with k2:
        tarjeta("Tasa de aprovechamiento", barra(100 * blancos / total), "Fracción al contenedor blanco")
    with k3:
        tarjeta("Certidumbre media", f"{stats['certidumbre_media']:.1f}%",
                f"Rango del modelo: {CERTIDUMBRE_MIN:.0f}–{CERTIDUMBRE_MAX:.0f}%")
    with k4:
        tarjeta("Último evento", html.escape(str(stats["ultima"])[11:19] or "—"),
                html.escape(str(stats["ultima"])[:10]))

    # --- Fila 1: proporción por color + volumen por material ------------------
    c1, c2 = st.columns([1, 1.6], gap="medium")
    with c1, st.container(key="tarjeta_dona"):
        fig = px.pie(por_color, names="Contenedor", values="cantidad", hole=0.6,
                     color="Contenedor", color_discrete_map=COLOR_POR_ETIQUETA,
                     category_orders={"Contenedor": ORDEN_ETIQUETAS})
        fig.update_traces(marker_line_color=BORDE_MARCAS, marker_line_width=1,
                          textinfo="percent", sort=False,
                          hovertemplate="%{label}<br>%{value} eventos (%{percent})<extra></extra>")
        fig.update_layout(annotations=[{"text": f"<b>{total}</b><br>eventos", "showarrow": False,
                                        "font": {"size": 16}}])
        st.plotly_chart(_estilo(fig, "Distribución por contenedor (Res. 2184)"), width="stretch")

    with c2, st.container(key="tarjeta_material"):
        por_mat = _aplicar_etiquetas(stats["por_material"]).sort_values("cantidad")
        fig = px.bar(por_mat, x="cantidad", y="material", orientation="h", color="Contenedor",
                     color_discrete_map=COLOR_POR_ETIQUETA,
                     category_orders={"Contenedor": ORDEN_ETIQUETAS},
                     custom_data=["certidumbre_media"],
                     labels={"cantidad": "Eventos", "material": ""})
        fig.update_traces(marker_line_color=BORDE_MARCAS, marker_line_width=1,
                          hovertemplate="%{y}<br>%{x} eventos · certidumbre media "
                                        "%{customdata[0]:.1f}%<extra></extra>")
        fig.update_layout(barcornerradius=4, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(_estilo(fig, "Volumen por clase de residuo"), width="stretch")

    # --- Fila 2: serie temporal + distribución de certidumbre -----------------
    c3, c4 = st.columns([1.6, 1], gap="medium")
    with c3, st.container(key="tarjeta_serie"):
        serie = _aplicar_etiquetas(stats["serie"])
        fig = px.bar(serie, x="periodo", y="cantidad", color="Contenedor",
                     color_discrete_map=COLOR_POR_ETIQUETA,
                     category_orders={"Contenedor": ORDEN_ETIQUETAS},
                     labels={"periodo": "", "cantidad": "Eventos"})
        fig.update_traces(marker_line_color=BORDE_MARCAS, marker_line_width=0.5)
        fig.update_layout(barmode="stack", bargap=0.15)
        st.plotly_chart(_estilo(fig, f"Eventos por {granularidad}"), width="stretch")

    with c4, st.container(key="tarjeta_certidumbre"):
        cert = _aplicar_etiquetas(stats["certidumbres"])
        fig = px.box(cert, x="Contenedor", y="certidumbre", color="Contenedor",
                     color_discrete_map=COLOR_POR_ETIQUETA,
                     category_orders={"Contenedor": ORDEN_ETIQUETAS},
                     labels={"certidumbre": "Certidumbre (%)", "Contenedor": ""})
        fig.update_traces(marker_line_color=BORDE_MARCAS, line_color=BORDE_MARCAS)
        fig.update_layout(showlegend=False)
        st.plotly_chart(_estilo(fig, "Certidumbre por contenedor"), width="stretch")

    # --- Tabla de eventos recientes ------------------------------------------
    with st.container(key="tarjeta_tabla"):
        st.markdown("**🕘 Últimos eventos registrados**")
        st.dataframe(
            db.get_events(100), hide_index=True, width="stretch", height=300,
            column_config={
                "id": "ID",
                "fecha_hora": st.column_config.DatetimeColumn("Fecha y hora", format="DD/MM/YYYY HH:mm:ss"),
                "material": "Clase IA",
                "contenedor": "Contenedor destino",
                "certidumbre": st.column_config.ProgressColumn("Certidumbre", min_value=0,
                                                               max_value=100, format="%.1f%%"),
            },
        )


def pantalla_dashboard() -> None:
    encabezado("📊 Central Analítica",
               "Métricas agrupadas por el código de colores de la Resolución 2184 de 2019 "
               "(MinAmbiente). Lecturas WAL concurrentes con la escritura del hilo de IA.")
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    granularidad = f1.selectbox("Agrupar serie por", ["segundo", "minuto", "hora", "dia"], index=1)
    ventana = f2.selectbox("Ventana de tiempo", ["Todo", "Última hora", "Últimas 24 h", "Últimos 7 días"])
    auto = f3.toggle("Auto-refresco (3 s)", value=True)
    with f4.popover("🗑️ Reiniciar datos", width="stretch"):
        st.warning("Se borrarán todos los eventos del escáner Edge. Esta acción no se puede deshacer.")
        if st.button("Confirmar borrado", type="primary"):
            n = db.reset_eventos()
            st.toast(f"Se eliminaron {n} eventos.")

    st.fragment(_tablero, run_every=3.0 if auto else None)(granularidad, ventana)


# ===========================================================================
# 7. ORQUESTACIÓN
# ===========================================================================
def barra_lateral(registrador: db.RegistradorAsincrono) -> tuple[str, float, float, bool]:
    with st.sidebar:
        st.markdown("## ♻️ Reciclaje Inteligente")
        st.caption("Pipeline Edge · NVIDIA Jetson")
        pantalla = st.radio("Navegación", [PANTALLA_ESCANER, PANTALLA_DASHBOARD],
                            label_visibility="collapsed")
        st.divider()
        st.markdown("**⚙️ Parámetros del escáner**")
        eventos_por_s = st.slider("Eventos registrados por segundo", 0.5, 10.0, 2.0, 0.5,
                                  help="Tasa máxima a la que el hilo de IA encola eventos para SQLite.")
        umbral = st.slider("Certidumbre mínima para registrar (%)", CERTIDUMBRE_MIN, CERTIDUMBRE_MAX,
                           CERTIDUMBRE_MIN, 1.0)
        hud = st.toggle("Mostrar métricas sobre el video", value=True)
        st.divider()
        st.markdown("**🗂️ Taxonomía activa**")
        for clase in TAXONOMIA:
            st.markdown(f"<small><span class='chip-dot' style='display:inline-block;vertical-align:middle;"
                        f"background:{COLOR_HEX[clase.color]}'></span>&nbsp;{html.escape(clase.nombre)}</small>",
                        unsafe_allow_html=True)
        st.divider()
        r = registrador.metricas()
        estado = "🟢 activo" if r["hilo_vivo"] else "🔴 detenido"
        st.caption(f"Hilo escritor SQLite: {estado} · escritos {r['escritos']} · "
                   f"descartados {r['descartados']} · BD: `{db.RUTA_BD.name}` (WAL)")
        if r["ultimo_error"]:
            st.caption(f"⚠️ Último error de escritura: {r['ultimo_error']}")
    return pantalla, eventos_por_s, umbral, hud


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    modelo = cargar_modelo()
    registrador = cargar_registrador()
    if not registrador.vivo:          # Recupera el hilo escritor si murió
        cargar_registrador.clear()
        registrador = cargar_registrador()

    pantalla, eventos_por_s, umbral, hud = barra_lateral(registrador)
    if pantalla == PANTALLA_ESCANER:
        pantalla_escaner(modelo, registrador, eventos_por_s, umbral, hud)
    else:
        pantalla_dashboard()


main()
