"""
database.py — Capa de Datos del Sistema de Reciclaje Inteligente
=================================================================
Persistencia local en SQLite de los eventos de clasificación.

Tabla principal: `clasificaciones`
    id            INTEGER  PK autoincremental
    fecha_hora    TEXT     'YYYY-MM-DD HH:MM:SS' (hora local)
    material      TEXT     PET, PEAD, Vidrio, Metal, Papel/Cartón, No aprovechable, ...
    confianza     REAL     0-100 (%)
    contenedor    TEXT     Blanco / Negro / Verde (Res. 2184 de 2019)
    -- Columnas de trazabilidad (opcionales, no alteran el esquema solicitado):
    origen        TEXT     'camara', 'trashnet', 'taco', ... (permite filtrar el tablero)
    motor         TEXT     Motor de visión que produjo la clasificación
    etiqueta_real TEXT     Etiqueta verdadera (solo datasets públicos -> exactitud)
    archivo       TEXT     Imagen de origen (hace idempotente la ingesta masiva)

Decisiones de arquitectura (auditoría):
    1. SIN dependencia de Streamlit. La versión anterior usaba @st.cache_resource
       para compartir UNA conexión entre hilos protegida por un Lock. Eso no
       funciona fuera del hilo del script (hilo de inferencia de WebRTC, script
       `ingesta_publica.py`) y serializaba lecturas y escrituras.
    2. Una conexión corta por operación. Abrir SQLite cuesta microsegundos y
       cada hilo/proceso queda aislado: no hace falta `check_same_thread=False`
       ni candados globales.
    3. Modo WAL + busy_timeout. Los lectores (tablero) no bloquean al escritor
       (cámara / ingesta) y viceversa; si dos procesos escriben a la vez, SQLite
       espera en lugar de lanzar "database is locked".
    4. Agregaciones en SQL (GROUP BY) para que el tablero soporte decenas de
       miles de filas sin cargar toda la tabla en Pandas.
    5. Migración automática desde la tabla antigua `eventos_reciclaje`.

Autor: Johan Lasso
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
# Se puede sobreescribir con la variable de entorno RECICLAJE_DB (útil en pruebas).
RUTA_BD: Path = Path(os.environ.get("RECICLAJE_DB", Path(__file__).parent / "data" / "reciclaje.db"))

TABLA = "clasificaciones"
FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"

# Política de registro: solo se guardan detecciones con confianza ESTRICTAMENTE mayor.
UMBRAL_REGISTRO = 60.0

# Contenedor -> tipo de residuo (para el gráfico de dona).
TIPO_POR_CONTENEDOR = {
    "Blanco": "Aprovechable",
    "Negro": "No aprovechable",
    "Verde": "Orgánico",
}
CONTENEDORES_VALIDOS = tuple(TIPO_POR_CONTENEDOR)

# Material del sistema -> grupo comparable con las etiquetas de TrashNet/TACO.
# (PET y PEAD se evalúan juntos como "Plástico": los datasets no los distinguen.)
GRUPO_VALIDACION = {
    "PET": "Plástico",
    "PEAD": "Plástico",
    "Vidrio": "Vidrio",
    "Metal": "Metal",
    "Papel/Cartón": "Papel/Cartón",
    "No aprovechable": "No aprovechable",
    "No identificado": "No identificado",
}


# Clases normativas finas de `ingesta_publica.py` -> grupo comparable.
GRUPO_VALIDACION.update({"Plástico": "Plástico", "Papel": "Papel/Cartón", "Cartón": "Papel/Cartón"})


def grupo_material(material: str) -> str:
    """
    Reduce un material o clase normativa a su grupo de validación.

    Ignora el sufijo entre paréntesis de las clases finas, de modo que
    «No aprovechable (Multicapa)» y «No aprovechable (Icopor)» se comparan como
    «No aprovechable», «Metal (Blanco)» como «Metal» y «Cartón (Tetra Pak)» como
    «Papel/Cartón».
    """
    if not material:
        return material
    base = re.sub(r"\s*\([^)]*\)\s*$", "", str(material)).strip()
    return GRUPO_VALIDACION.get(base, base)

# Columnas que se pueden escribir/actualizar (lista blanca contra inyección SQL
# en nombres de columna, que no se pueden parametrizar con "?").
_COLUMNAS_ESCRIBIBLES = (
    "fecha_hora", "material", "confianza", "contenedor",
    "origen", "motor", "etiqueta_real", "archivo",
)

_SQL_ESQUEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLA} (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_hora    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
    material      TEXT    NOT NULL,
    confianza     REAL    NOT NULL CHECK (confianza >= 0 AND confianza <= 100),
    contenedor    TEXT    NOT NULL CHECK (contenedor IN ('Blanco', 'Negro', 'Verde')),
    origen        TEXT    NOT NULL DEFAULT 'camara',
    motor         TEXT,
    etiqueta_real TEXT,
    archivo       TEXT
);
CREATE INDEX IF NOT EXISTS idx_clasif_fecha    ON {TABLA}(fecha_hora);
CREATE INDEX IF NOT EXISTS idx_clasif_origen   ON {TABLA}(origen);
CREATE INDEX IF NOT EXISTS idx_clasif_material ON {TABLA}(material);
-- Evita duplicar una misma imagen de un dataset con el mismo motor
-- (la ingesta se puede relanzar sin inflar el tablero).
CREATE UNIQUE INDEX IF NOT EXISTS uq_clasif_archivo
    ON {TABLA}(origen, archivo, motor) WHERE archivo IS NOT NULL;
"""

_init_lock = threading.Lock()
_inicializada: set[str] = set()


# ---------------------------------------------------------------------------
# Conexión
# ---------------------------------------------------------------------------
def _abrir() -> sqlite3.Connection:
    RUTA_BD.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(RUTA_BD, timeout=10.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout = 10000")
    con.execute("PRAGMA synchronous = NORMAL")  # Seguro con WAL y mucho más rápido
    return con


@contextmanager
def conectar() -> Iterator[sqlite3.Connection]:
    """Conexión de vida corta: commit si todo sale bien, rollback si hay error."""
    init_db()
    con = _abrir()
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    """Crea el esquema (idempotente) y migra la tabla antigua. Se ejecuta una vez por proceso."""
    clave = str(RUTA_BD)
    if clave in _inicializada:
        return
    with _init_lock:
        if clave in _inicializada:
            return
        con = _abrir()
        try:
            con.execute("PRAGMA journal_mode = WAL")  # Persistente en el archivo
            con.executescript(_SQL_ESQUEMA)
            _migrar_tabla_antigua(con)
            con.commit()
        finally:
            con.close()
        _inicializada.add(clave)


def _migrar_tabla_antigua(con: sqlite3.Connection) -> None:
    """Copia `eventos_reciclaje` (esquema v1) a `clasificaciones` y la renombra."""
    existe = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='eventos_reciclaje'"
    ).fetchone()
    if not existe:
        return
    con.execute(
        f"""INSERT INTO {TABLA} (fecha_hora, material, confianza, contenedor, origen, motor)
            SELECT replace(substr(timestamp, 1, 19), 'T', ' '), material,
                   min(100.0, max(0.0, confianza)), contenedor, 'camara', 'legado v1'
            FROM eventos_reciclaje
            WHERE contenedor IN ('Blanco', 'Negro', 'Verde')"""
    )
    # Se conserva como respaldo en lugar de borrarla.
    con.execute("ALTER TABLE eventos_reciclaje RENAME TO eventos_reciclaje_migrada_v1")


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _normalizar_fecha(fecha: datetime | str | None) -> str:
    if fecha is None:
        return datetime.now().strftime(FORMATO_FECHA)
    if isinstance(fecha, datetime):
        return fecha.strftime(FORMATO_FECHA)
    return str(fecha).replace("T", " ")[:19]


def _validar_registro(material: str, confianza: float, contenedor: str) -> float:
    if not material:
        raise ValueError("El material no puede estar vacío.")
    if contenedor not in CONTENEDORES_VALIDOS:
        raise ValueError(f"Contenedor inválido: {contenedor!r}. Use {CONTENEDORES_VALIDOS}.")
    confianza = float(confianza)
    if not 0.0 <= confianza <= 100.0:
        raise ValueError(f"Confianza fuera de rango (0-100): {confianza}")
    return round(confianza, 2)


def _filtros(origenes: Iterable[str] | None = None,
             desde: datetime | str | None = None,
             hasta: datetime | str | None = None) -> tuple[str, list[Any]]:
    """Construye la cláusula WHERE parametrizada común a las consultas del tablero."""
    condiciones: list[str] = []
    params: list[Any] = []
    if origenes is not None:
        origenes = list(origenes)
        if not origenes:
            return " WHERE 0", []
        condiciones.append(f"origen IN ({', '.join('?' * len(origenes))})")
        params.extend(origenes)
    if desde is not None:
        condiciones.append("fecha_hora >= ?")
        params.append(_normalizar_fecha(desde))
    if hasta is not None:
        condiciones.append("fecha_hora <= ?")
        params.append(_normalizar_fecha(hasta))
    return (" WHERE " + " AND ".join(condiciones)) if condiciones else "", params


# ===========================================================================
# CRUD
# ===========================================================================
def crear_clasificacion(material: str, confianza: float, contenedor: str, *,
                        fecha_hora: datetime | str | None = None,
                        origen: str = "camara",
                        motor: str | None = None,
                        etiqueta_real: str | None = None,
                        archivo: str | None = None) -> int:
    """CREATE — Inserta una clasificación y devuelve su id."""
    confianza = _validar_registro(material, confianza, contenedor)
    with conectar() as con:
        cur = con.execute(
            f"""INSERT INTO {TABLA}
                (fecha_hora, material, confianza, contenedor, origen, motor, etiqueta_real, archivo)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (_normalizar_fecha(fecha_hora), material, confianza, contenedor,
             origen, motor, etiqueta_real, archivo),
        )
        return int(cur.lastrowid)


def crear_lote(registros: Iterable[Mapping[str, Any]]) -> int:
    """
    CREATE masivo — Inserta muchos registros en UNA transacción (cientos de veces
    más rápido que un commit por fila). Ignora duplicados (origen, archivo, motor).

    Returns:
        Número de filas realmente insertadas.
    """
    filas = []
    for r in registros:
        conf = _validar_registro(r["material"], r["confianza"], r["contenedor"])
        filas.append((
            _normalizar_fecha(r.get("fecha_hora")), r["material"], conf, r["contenedor"],
            r.get("origen", "camara"), r.get("motor"), r.get("etiqueta_real"), r.get("archivo"),
        ))
    if not filas:
        return 0
    with conectar() as con:
        antes = con.total_changes
        con.executemany(
            f"""INSERT OR IGNORE INTO {TABLA}
                (fecha_hora, material, confianza, contenedor, origen, motor, etiqueta_real, archivo)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            filas,
        )
        return con.total_changes - antes


def obtener_clasificacion(id_registro: int) -> dict[str, Any] | None:
    """READ — Devuelve un registro por id (o None)."""
    with conectar() as con:
        fila = con.execute(f"SELECT * FROM {TABLA} WHERE id = ?", (int(id_registro),)).fetchone()
    return dict(fila) if fila else None


def listar_clasificaciones(limite: int | None = None, *,
                           origenes: Iterable[str] | None = None,
                           desde: datetime | str | None = None,
                           hasta: datetime | str | None = None) -> pd.DataFrame:
    """READ — Registros como DataFrame, del más reciente al más antiguo."""
    where, params = _filtros(origenes, desde, hasta)
    sql = f"SELECT * FROM {TABLA}{where} ORDER BY id DESC"
    if limite is not None:
        sql += " LIMIT ?"
        params.append(int(limite))
    with conectar() as con:
        df = pd.read_sql_query(sql, con, params=params)
    if not df.empty:
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
    return df


def actualizar_clasificacion(id_registro: int, **campos: Any) -> bool:
    """UPDATE — Modifica columnas permitidas. Devuelve True si el registro existía."""
    desconocidas = set(campos) - set(_COLUMNAS_ESCRIBIBLES)
    if desconocidas:
        raise ValueError(f"Columnas no permitidas: {sorted(desconocidas)}")
    if not campos:
        return False
    actual = obtener_clasificacion(id_registro)
    if actual is None:
        return False
    actual.update(campos)
    campos["confianza"] = _validar_registro(actual["material"], actual["confianza"], actual["contenedor"])
    if "fecha_hora" in campos:
        campos["fecha_hora"] = _normalizar_fecha(campos["fecha_hora"])
    asignaciones = ", ".join(f"{col} = ?" for col in campos)
    with conectar() as con:
        cur = con.execute(f"UPDATE {TABLA} SET {asignaciones} WHERE id = ?",
                          (*campos.values(), int(id_registro)))
        return cur.rowcount > 0


def eliminar_clasificacion(id_registro: int) -> bool:
    """DELETE — Borra un registro por id."""
    with conectar() as con:
        return con.execute(f"DELETE FROM {TABLA} WHERE id = ?", (int(id_registro),)).rowcount > 0


def eliminar_por_origen(origen: str) -> int:
    """DELETE — Borra todos los registros de un origen (p. ej. deshacer una ingesta)."""
    with conectar() as con:
        return con.execute(f"DELETE FROM {TABLA} WHERE origen = ?", (origen,)).rowcount


def reiniciar() -> int:
    """DELETE — Vacía la tabla (botón 'Reiniciar datos')."""
    with conectar() as con:
        return con.execute(f"DELETE FROM {TABLA}").rowcount


# ===========================================================================
# CONSULTAS ANALÍTICAS (alimentan el Tablero Estadístico)
# ===========================================================================
def firma_datos() -> tuple[int, int]:
    """(total, id máximo). Cambia con cada escritura -> clave de caché del tablero."""
    with conectar() as con:
        total, max_id = con.execute(f"SELECT COUNT(*), COALESCE(MAX(id), 0) FROM {TABLA}").fetchone()
    return int(total), int(max_id)


def origenes_disponibles() -> list[str]:
    with conectar() as con:
        return [f[0] for f in con.execute(f"SELECT DISTINCT origen FROM {TABLA} ORDER BY origen")]


def resumen_kpis(origenes: Iterable[str] | None = None) -> dict[str, Any]:
    where, params = _filtros(origenes)
    with conectar() as con:
        fila = con.execute(
            f"""SELECT COUNT(*) AS total,
                       AVG(confianza) AS confianza_promedio,
                       SUM(contenedor = 'Blanco') AS aprovechables,
                       MIN(fecha_hora) AS primera, MAX(fecha_hora) AS ultima
                FROM {TABLA}{where}""", params).fetchone()
    return dict(fila)


def conteo_por_material(origenes: Iterable[str] | None = None) -> pd.DataFrame:
    """Volumen por material (columnas: material, contenedor, tipo, cantidad, confianza_media)."""
    where, params = _filtros(origenes)
    with conectar() as con:
        df = pd.read_sql_query(
            f"""SELECT material, contenedor, COUNT(*) AS cantidad,
                       ROUND(AVG(confianza), 1) AS confianza_media
                FROM {TABLA}{where}
                GROUP BY material, contenedor ORDER BY cantidad DESC""", con, params=params)
    df["tipo"] = df["contenedor"].map(TIPO_POR_CONTENEDOR)
    return df


def proporcion_aprovechable(origenes: Iterable[str] | None = None) -> pd.DataFrame:
    """Proporción reciclable vs. no aprovechable (columnas: tipo, cantidad)."""
    where, params = _filtros(origenes)
    with conectar() as con:
        df = pd.read_sql_query(
            f"SELECT contenedor, COUNT(*) AS cantidad FROM {TABLA}{where} GROUP BY contenedor",
            con, params=params)
    df["tipo"] = df["contenedor"].map(TIPO_POR_CONTENEDOR)
    return df.groupby("tipo", as_index=False)["cantidad"].sum()


def volumen_por_dia(origenes: Iterable[str] | None = None) -> pd.DataFrame:
    """Volumen diario por tipo (columnas: dia, tipo, cantidad)."""
    where, params = _filtros(origenes)
    with conectar() as con:
        df = pd.read_sql_query(
            f"""SELECT substr(fecha_hora, 1, 10) AS dia, contenedor, COUNT(*) AS cantidad
                FROM {TABLA}{where} GROUP BY dia, contenedor ORDER BY dia""", con, params=params)
    df["tipo"] = df["contenedor"].map(TIPO_POR_CONTENEDOR)
    df["dia"] = pd.to_datetime(df["dia"])
    return df.groupby(["dia", "tipo"], as_index=False)["cantidad"].sum()


def datos_validacion(origenes: Iterable[str] | None = None) -> pd.DataFrame:
    """Pares (etiqueta_real, material) de registros con verdad de terreno (datasets públicos)."""
    where, params = _filtros(origenes)
    where = (where + " AND " if where else " WHERE ") + "etiqueta_real IS NOT NULL"
    with conectar() as con:
        return pd.read_sql_query(
            f"""SELECT origen, motor, etiqueta_real, material, COUNT(*) AS cantidad
                FROM {TABLA}{where} GROUP BY origen, motor, etiqueta_real, material""",
            con, params=params)


# ===========================================================================
# Compatibilidad con la API v1 (app/scripts anteriores)
# ===========================================================================
def insert_evento(material: str, confianza: float, contenedor: str) -> int:
    return crear_clasificacion(material, confianza, contenedor)


def get_estadisticas() -> pd.DataFrame:
    df = listar_clasificaciones()
    return df.rename(columns={"fecha_hora": "timestamp"})[
        ["id", "timestamp", "material", "confianza", "contenedor"]]


def reset_db() -> None:
    reiniciar()


# ===========================================================================
# API del escáner Edge (vision_edge.py / app_main.py)
# ===========================================================================
# `app_main.py` y `vision_edge.py` se escribieron contra una versión de este módulo
# que exponía un registrador asíncrono. Se restaura aquí sobre la MISMA tabla
# `clasificaciones`, con origen = ORIGEN_EDGE para poder filtrar y borrar sus eventos.
import queue as _queue  # noqa: E402
import unicodedata as _unicodedata  # noqa: E402

ORIGEN_EDGE = "camara_edge"
_COLOR_A_CONTENEDOR = {"BLANCO": "Blanco", "NEGRO": "Negro", "VERDE": "Verde"}
_FORMATO_PERIODO = {"segundo": "%Y-%m-%d %H:%M:%S", "minuto": "%Y-%m-%d %H:%M",
                    "hora": "%Y-%m-%d %H:00", "dia": "%Y-%m-%d"}


def color_de_contenedor(destino: str) -> str:
    """
    Color canónico en MAYÚSCULAS a partir de cualquier rótulo de contenedor.

    >>> color_de_contenedor("BLANCO (Aprovechable)")
    'BLANCO'
    >>> color_de_contenedor("Verde")
    'VERDE'
    """
    ascii_ = _unicodedata.normalize("NFKD", str(destino)).encode("ascii", "ignore").decode()
    primera = (ascii_.strip().split() or [""])[0].upper()
    if primera not in _COLOR_A_CONTENEDOR:
        raise ValueError(f"Contenedor desconocido: {destino!r}")
    return primera


def contenedor_canonico(destino: str) -> str:
    """'NEGRO (No Aprovechable ...)' -> 'Negro' (valor válido para la tabla)."""
    return _COLOR_A_CONTENEDOR[color_de_contenedor(destino)]


class RegistradorAsincrono:
    """
    Hilo escritor único: el hilo de inferencia encola sin bloquear y este hilo
    inserta por lotes en SQLite (WAL). Si la cola se llena, el evento se descarta
    (nunca se frena el video por el disco).
    """

    def __init__(self, capacidad: int = 2000, lote: int = 50, origen: str = ORIGEN_EDGE,
                 motor: str | None = None) -> None:
        self._cola: _queue.Queue[dict[str, Any]] = _queue.Queue(maxsize=capacidad)
        self._lote = lote
        self.origen = origen
        self.motor = motor
        self._escritos = 0
        self._descartados = 0
        self._ultimo_error: str | None = None
        self._detener = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, name="sqlite-escritor", daemon=True)
        self._hilo.start()

    @property
    def vivo(self) -> bool:
        return self._hilo.is_alive()

    def encolar_deteccion(self, material: str, contenedor: str, certidumbre: float) -> bool:
        """No bloqueante. Devuelve False si el evento se descartó (cola llena o datos inválidos)."""
        try:
            registro = {"material": material, "contenedor": contenedor_canonico(contenedor),
                        "confianza": max(0.0, min(100.0, float(certidumbre))),
                        "fecha_hora": _normalizar_fecha(None), "origen": self.origen,
                        "motor": self.motor}
            self._cola.put_nowait(registro)
            return True
        except (_queue.Full, ValueError, KeyError):
            self._descartados += 1
            return False

    def metricas(self) -> dict[str, Any]:
        return {"escritos": self._escritos, "pendientes": self._cola.qsize(),
                "descartados": self._descartados, "hilo_vivo": self.vivo,
                "ultimo_error": self._ultimo_error}

    def detener(self, timeout: float = 2.0) -> None:
        self._detener.set()
        self._hilo.join(timeout)

    def _bucle(self) -> None:
        while not (self._detener.is_set() and self._cola.empty()):
            try:
                lote = [self._cola.get(timeout=0.5)]
            except _queue.Empty:
                continue
            while len(lote) < self._lote:
                try:
                    lote.append(self._cola.get_nowait())
                except _queue.Empty:
                    break
            try:
                self._escritos += crear_lote(lote)
                self._ultimo_error = None
            except Exception as exc:  # El hilo nunca debe morir en silencio
                self._ultimo_error = f"{type(exc).__name__}: {exc}"


_registrador: RegistradorAsincrono | None = None
_registrador_lock = threading.Lock()


def obtener_registrador(motor: str | None = None) -> RegistradorAsincrono:
    """Un registrador por proceso (se recrea si su hilo murió)."""
    global _registrador
    with _registrador_lock:
        if _registrador is None or not _registrador.vivo:
            _registrador = RegistradorAsincrono(motor=motor)
        elif motor:
            _registrador.motor = motor
        return _registrador


def get_stats(granularidad: str = "minuto", desde: datetime | str | None = None) -> dict[str, Any]:
    """Agregados del tablero Edge (solo eventos con origen ORIGEN_EDGE)."""
    formato = _FORMATO_PERIODO.get(granularidad, _FORMATO_PERIODO["minuto"])
    where, params = _filtros([ORIGEN_EDGE], desde=desde)
    color = ("CASE contenedor WHEN 'Blanco' THEN 'BLANCO' WHEN 'Negro' THEN 'NEGRO' "
             "ELSE 'VERDE' END")
    with conectar() as con:
        kpi = con.execute(f"SELECT COUNT(*), AVG(confianza), MAX(fecha_hora) FROM {TABLA}{where}",
                          params).fetchone()
        leer = lambda sql: pd.read_sql_query(sql, con, params=params)  # noqa: E731
        por_contenedor = leer(f"SELECT {color} AS color, COUNT(*) AS cantidad FROM {TABLA}{where} "
                              "GROUP BY color")
        por_material = leer(f"SELECT material, {color} AS color, COUNT(*) AS cantidad, "
                            f"AVG(confianza) AS certidumbre_media FROM {TABLA}{where} "
                            "GROUP BY material, color")
        serie = leer(f"SELECT strftime('{formato}', fecha_hora) AS periodo, {color} AS color, "
                     f"COUNT(*) AS cantidad FROM {TABLA}{where} GROUP BY periodo, color "
                     "ORDER BY periodo")
        certidumbres = leer(f"SELECT {color} AS color, confianza AS certidumbre FROM {TABLA}{where} "
                            "ORDER BY id DESC LIMIT 5000")
    return {"total": kpi[0] or 0, "certidumbre_media": float(kpi[1] or 0.0), "ultima": kpi[2] or "",
            "por_contenedor": por_contenedor, "por_material": por_material, "serie": serie,
            "certidumbres": certidumbres}


def get_events(limite: int = 100) -> pd.DataFrame:
    """Últimos eventos del escáner Edge."""
    with conectar() as con:
        return pd.read_sql_query(
            f"SELECT id, fecha_hora, material, contenedor, confianza AS certidumbre FROM {TABLA} "
            "WHERE origen = ? ORDER BY id DESC LIMIT ?", con, params=[ORIGEN_EDGE, int(limite)])


def reset_eventos() -> int:
    """Borra solo los eventos del escáner Edge."""
    return eliminar_por_origen(ORIGEN_EDGE)
