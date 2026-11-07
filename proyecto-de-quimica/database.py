"""
database.py — Capa de Datos del Sistema de Reciclaje Inteligente
=================================================================
Persistencia local en SQLite (modo WAL) de los eventos de clasificación.

Este módulo atiende a DOS interfaces que conviven sobre el mismo archivo .db:

A) Pipeline Edge (`vision_edge.py` + `app_main.py`) — tabla `clasificaciones_residuos`
    id            INTEGER  PK autoincremental
    fecha_hora    TEXT     'YYYY-MM-DD HH:MM:SS.mmm' (hora local)
    material      TEXT     Clase IA de la taxonomía colombiana (p. ej. "Envase Multicapa (Tetra Pak)")
    contenedor    TEXT     "BLANCO (Aprovechable)" / "NEGRO (No Aprovechable - ...)" / "VERDE (Orgánico)"
    certidumbre   REAL     0-100 (%)
  API: init_db(), log_event(), get_stats(), get_events(), reset_eventos(),
       RegistradorAsincrono / obtener_registrador()  (cola + hilo escritor por lotes)

B) Aplicación clásica (`app.py`, `ingesta_publica.py`) — tabla `clasificaciones`
    (id, fecha_hora, material, confianza, contenedor, origen, motor, etiqueta_real, archivo)
  API CRUD y analítica intacta: crear_clasificacion(), listar_clasificaciones(), ...

Estrategia de concurrencia (aplica a ambas interfaces):
    1. SIN dependencia de Streamlit: el módulo se usa desde el hilo del script,
       desde el hilo de inferencia de WebRTC, desde el hilo escritor y desde
       scripts por consola. Ningún objeto de Streamlit cruza esos hilos.
    2. Una conexión de vida corta por operación. `sqlite3.Connection` no debe
       compartirse entre hilos; abrir una conexión SQLite cuesta microsegundos,
       así que cada operación abre la suya y la cierra. No se necesita
       `check_same_thread=False` ni un candado global que serialice lecturas.
    3. `PRAGMA journal_mode=WAL` (persistente en el archivo) + `synchronous=NORMAL`.
       Con WAL los lectores leen una instantánea consistente mientras el escritor
       agrega páginas al log: el tablero NUNCA bloquea al hilo de IA y viceversa.
       NORMAL solo hace fsync en los checkpoints: es seguro ante caídas de la
       aplicación (no del sistema operativo) y multiplica el rendimiento de escritura.
    4. `busy_timeout` + bloqueo iterativo: si aun así aparece "database is locked"
       (p. ej. un checkpoint o dos procesos escribiendo), la operación se reintenta
       con espera exponencial acotada en lugar de propagar el error.
    5. Un ÚNICO hilo escritor (RegistradorAsincrono) consume una cola y agrupa los
       eventos en transacciones por lotes: decenas de eventos/segundo cuestan una
       sola transacción, y SQLite nunca ve escritores concurrentes desde este proceso.
    6. Agregaciones en SQL (GROUP BY) para que el tablero escale a cientos de miles
       de filas sin cargar la tabla completa en Pandas.

Autor: Johan Lasso
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, TypeVar

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


def grupo_material(material: str) -> str:
    return GRUPO_VALIDACION.get(material, material)

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

-- Tabla del pipeline Edge (vision_edge.py / app_main.py).
-- El CHECK sobre `contenedor` impide que un error de mapeo introduzca un color
-- que no existe en la Resolución 2184 de 2019.
CREATE TABLE IF NOT EXISTS clasificaciones_residuos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_hora  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now', 'localtime')),
    material    TEXT    NOT NULL,
    contenedor  TEXT    NOT NULL CHECK (
                    contenedor LIKE 'BLANCO%' OR contenedor LIKE 'NEGRO%' OR contenedor LIKE 'VERDE%'),
    certidumbre REAL    NOT NULL CHECK (certidumbre >= 0 AND certidumbre <= 100)
);
CREATE INDEX IF NOT EXISTS idx_residuos_fecha      ON clasificaciones_residuos(fecha_hora);
CREATE INDEX IF NOT EXISTS idx_residuos_contenedor ON clasificaciones_residuos(contenedor);
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
    """
    Crea el esquema (idempotente), activa WAL y migra la tabla antigua.

    Se ejecuta una sola vez por proceso y por archivo de base de datos. Usa el
    patrón "double-checked locking": la primera comprobación (sin candado) hace
    que las llamadas posteriores cuesten un lookup en un set; la segunda (con
    candado) evita que dos hilos que arrancan a la vez creen el esquema dos veces.

    `journal_mode=WAL` es persistente: queda grabado en el archivo .db, por lo
    que cualquier conexión futura (incluso de otro proceso) trabajará en WAL.
    `synchronous=NORMAL`, en cambio, es por conexión y por eso se fija en `_abrir()`.
    """
    clave = str(RUTA_BD)
    if clave in _inicializada:
        return
    with _init_lock:
        if clave in _inicializada:
            return
        con = _abrir()
        try:
            modo = con.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(modo).lower() != "wal":
                # Ocurre en sistemas de archivos sin memoria compartida (algunos
                # montajes de red). Se registra en vez de fallar: la app sigue
                # funcionando, pero con lectores y escritor serializados.
                logging.getLogger(__name__).warning(
                    "SQLite no pudo activar WAL (modo actual: %s). Las lecturas del "
                    "tablero podrán bloquearse durante las escrituras.", modo)
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
# PIPELINE EDGE — tabla `clasificaciones_residuos`
# (consumida por vision_edge.py y app_main.py)
# ===========================================================================
TABLA_EDGE = "clasificaciones_residuos"

# Los tres colores de la Resolución 2184 de 2019. El texto completo del destino
# ("BLANCO (Aprovechable)") se guarda tal cual; el color se deriva de su prefijo.
COLORES_RES_2184 = ("BLANCO", "NEGRO", "VERDE")

# Bloqueo iterativo: busy_timeout corto por intento + reintentos con espera
# exponencial. Con un busy_timeout de 10 s (el de la API clásica) un tablero
# bloqueado congelaría la interfaz; aquí se prefieren intentos cortos y acotados.
_BUSY_TIMEOUT_EDGE_MS = 1500
_REINTENTOS = 6
_ESPERA_INICIAL_S = 0.02
_ESPERA_MAXIMA_S = 0.5

# Expresión SQL que extrae el color (BLANCO/NEGRO/VERDE) de la columna contenedor.
_SQL_COLOR = (
    "CASE WHEN contenedor LIKE 'BLANCO%' THEN 'BLANCO' "
    "WHEN contenedor LIKE 'NEGRO%' THEN 'NEGRO' "
    "WHEN contenedor LIKE 'VERDE%' THEN 'VERDE' ELSE 'OTRO' END"
)

_log = logging.getLogger(__name__)
T = TypeVar("T")


def color_de_contenedor(contenedor: str) -> str:
    """'BLANCO (Aprovechable)' -> 'BLANCO'. Lanza ValueError si no es un color válido."""
    color = str(contenedor).strip().split(" ", 1)[0].upper()
    if color not in COLORES_RES_2184:
        raise ValueError(f"Contenedor fuera de la Res. 2184 de 2019: {contenedor!r}")
    return color


def _ahora_ms() -> str:
    """Marca de tiempo local con milisegundos (varias escrituras por segundo)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _es_bloqueo(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, sqlite3.OperationalError) and ("locked" in msg or "busy" in msg)


def _con_reintentos(operacion: Callable[[], T], *, descripcion: str) -> T:
    """
    Bloqueo iterativo.

    Ejecuta `operacion` y, si SQLite responde "database is locked/busy", la
    reintenta con espera exponencial + jitter (0,02 s, 0,04 s, ... hasta 0,5 s).
    El jitter evita que dos hilos que chocaron vuelvan a chocar en sincronía.
    Cualquier otro error (esquema, CHECK, disco lleno) se propaga de inmediato:
    reintentarlo no lo arreglaría y ocultaría el problema.
    """
    espera = _ESPERA_INICIAL_S
    for intento in range(1, _REINTENTOS + 1):
        try:
            return operacion()
        except sqlite3.OperationalError as exc:
            if not _es_bloqueo(exc) or intento == _REINTENTOS:
                raise
            _log.debug("%s: base ocupada (intento %d/%d), reintentando en %.3f s",
                       descripcion, intento, _REINTENTOS, espera)
            time.sleep(espera * (0.5 + random.random()))
            espera = min(espera * 2, _ESPERA_MAXIMA_S)
    raise RuntimeError("inalcanzable")  # pragma: no cover


@contextmanager
def _conexion_edge() -> Iterator[sqlite3.Connection]:
    """Conexión corta con busy_timeout reducido (el reintento lo hace _con_reintentos)."""
    init_db()
    con = _abrir()
    con.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_EDGE_MS}")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


@dataclass(frozen=True, slots=True)
class EventoClasificacion:
    """
    Evento inmutable que viaja del hilo de IA al hilo escritor.

    Es `frozen`: una vez creado nadie puede modificarlo, así que cruzar hilos
    por la cola no requiere copias ni candados. La marca de tiempo se toma al
    CREAR el evento (momento de la detección), no al escribirlo en disco, para
    que el agrupamiento por lotes no distorsione la serie temporal.
    """
    material: str
    contenedor: str
    certidumbre: float
    fecha_hora: str

    @classmethod
    def crear(cls, material: str, contenedor: str, certidumbre: float,
              fecha_hora: datetime | str | None = None) -> "EventoClasificacion":
        if not material:
            raise ValueError("El material no puede estar vacío.")
        color_de_contenedor(contenedor)                  # valida el destino
        certidumbre = float(certidumbre)
        if not 0.0 <= certidumbre <= 100.0:
            raise ValueError(f"Certidumbre fuera de rango (0-100): {certidumbre}")
        if fecha_hora is None:
            marca = _ahora_ms()
        elif isinstance(fecha_hora, datetime):
            marca = fecha_hora.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        else:
            marca = str(fecha_hora).replace("T", " ")[:23]
        return cls(material, contenedor, round(certidumbre, 2), marca)

    def como_fila(self) -> tuple[str, str, str, float]:
        return (self.fecha_hora, self.material, self.contenedor, self.certidumbre)


_SQL_INSERT_EDGE = (f"INSERT INTO {TABLA_EDGE} (fecha_hora, material, contenedor, certidumbre) "
                    "VALUES (?, ?, ?, ?)")


def log_event(material: str, contenedor: str, certidumbre: float,
              fecha_hora: datetime | str | None = None) -> int:
    """
    Escritura SÍNCRONA de un evento (con bloqueo iterativo). Devuelve el id.

    Úsela desde scripts o pruebas. Desde el hilo de video NO se llama
    directamente: se usa `obtener_registrador().encolar(...)`, que nunca
    bloquea al hilo de inferencia por E/S de disco.
    """
    evento = EventoClasificacion.crear(material, contenedor, certidumbre, fecha_hora)

    def _insertar() -> int:
        with _conexion_edge() as con:
            return int(con.execute(_SQL_INSERT_EDGE, evento.como_fila()).lastrowid)

    return _con_reintentos(_insertar, descripcion="log_event")


def log_events(eventos: Iterable[EventoClasificacion]) -> int:
    """
    Escritura por LOTES en una sola transacción (la usa el hilo escritor).

    Una transacción por lote en vez de una por evento: con WAL cada commit
    agrega un marco al log; agrupar 50 eventos cuesta prácticamente lo mismo
    que escribir uno. Si el lote falla por bloqueo, se reintenta COMPLETO
    (la transacción hizo rollback, no quedan filas a medias).
    """
    filas = [e.como_fila() for e in eventos]
    if not filas:
        return 0

    def _insertar() -> int:
        with _conexion_edge() as con:
            con.executemany(_SQL_INSERT_EDGE, filas)
            return len(filas)

    return _con_reintentos(_insertar, descripcion=f"log_events[{len(filas)}]")


_GRANULARIDAD = {"segundo": 19, "minuto": 16, "hora": 13, "dia": 10}


def get_stats(granularidad: str = "minuto", desde: datetime | str | None = None) -> dict[str, Any]:
    """
    Lectura SEGURA de todas las métricas del tablero (con bloqueo iterativo).

    Todas las consultas se ejecutan dentro de UNA transacción de lectura
    (`BEGIN` explícito). En modo WAL eso fija una instantánea: aunque el hilo
    de IA siga escribiendo, los KPIs, el gráfico por contenedor y la serie
    temporal se calculan sobre exactamente las mismas filas (sin totales que
    "no cuadran" entre tarjetas). El lector nunca bloquea al escritor.

    Returns:
        dict con:
          total, certidumbre_media, primera, ultima          (escalares)
          por_contenedor  DataFrame[color, contenedor, cantidad, certidumbre_media]
          por_material    DataFrame[material, contenedor, color, cantidad, certidumbre_media]
          serie           DataFrame[periodo, color, cantidad]
          certidumbres    DataFrame[color, certidumbre]  (muestra para histograma/box)
    """
    if granularidad not in _GRANULARIDAD:
        raise ValueError(f"Granularidad inválida: {granularidad!r}. Use {tuple(_GRANULARIDAD)}")
    largo = _GRANULARIDAD[granularidad]
    where, params = "", []
    if desde is not None:
        where = " WHERE fecha_hora >= ?"
        params = [desde.strftime(FORMATO_FECHA) if isinstance(desde, datetime) else str(desde)]

    def _leer() -> dict[str, Any]:
        with _conexion_edge() as con:
            con.execute("BEGIN")   # instantánea consistente para todas las consultas
            kpi = con.execute(
                f"""SELECT COUNT(*) AS total, AVG(certidumbre) AS certidumbre_media,
                           MIN(fecha_hora) AS primera, MAX(fecha_hora) AS ultima
                    FROM {TABLA_EDGE}{where}""", params).fetchone()
            por_contenedor = pd.read_sql_query(
                f"""SELECT {_SQL_COLOR} AS color, contenedor, COUNT(*) AS cantidad,
                           ROUND(AVG(certidumbre), 2) AS certidumbre_media
                    FROM {TABLA_EDGE}{where}
                    GROUP BY contenedor ORDER BY cantidad DESC""", con, params=params)
            por_material = pd.read_sql_query(
                f"""SELECT material, contenedor, {_SQL_COLOR} AS color, COUNT(*) AS cantidad,
                           ROUND(AVG(certidumbre), 2) AS certidumbre_media
                    FROM {TABLA_EDGE}{where}
                    GROUP BY material, contenedor ORDER BY cantidad DESC""", con, params=params)
            serie = pd.read_sql_query(
                f"""SELECT substr(fecha_hora, 1, {largo}) AS periodo, {_SQL_COLOR} AS color,
                           COUNT(*) AS cantidad
                    FROM {TABLA_EDGE}{where}
                    GROUP BY periodo, color ORDER BY periodo""", con, params=params)
            # Muestra acotada (las 5 000 más recientes) para la distribución de certidumbre.
            certidumbres = pd.read_sql_query(
                f"""SELECT {_SQL_COLOR} AS color, certidumbre
                    FROM {TABLA_EDGE}{where} ORDER BY id DESC LIMIT 5000""", con, params=params)
        return {**dict(kpi), "por_contenedor": por_contenedor, "por_material": por_material,
                "serie": serie, "certidumbres": certidumbres}

    stats = _con_reintentos(_leer, descripcion="get_stats")
    if not stats["serie"].empty:
        stats["serie"]["periodo"] = pd.to_datetime(stats["serie"]["periodo"], errors="coerce")
    return stats


def get_events(limite: int = 200) -> pd.DataFrame:
    """Últimos eventos (más reciente primero) como DataFrame, con bloqueo iterativo."""
    def _leer() -> pd.DataFrame:
        with _conexion_edge() as con:
            return pd.read_sql_query(
                f"""SELECT id, fecha_hora, material, contenedor, certidumbre
                    FROM {TABLA_EDGE} ORDER BY id DESC LIMIT ?""", con, params=[int(limite)])

    df = _con_reintentos(_leer, descripcion="get_events")
    if not df.empty:
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
    return df


def reset_eventos() -> int:
    """Vacía la tabla del pipeline Edge (no toca la tabla `clasificaciones`)."""
    def _borrar() -> int:
        with _conexion_edge() as con:
            return con.execute(f"DELETE FROM {TABLA_EDGE}").rowcount

    return _con_reintentos(_borrar, descripcion="reset_eventos")


# ---------------------------------------------------------------------------
# Registrador asíncrono: cola acotada + UN hilo escritor por lotes
# ---------------------------------------------------------------------------
class RegistradorAsincrono:
    """
    Desacopla la inferencia (tiempo real) de la E/S de disco (latencia variable).

        hilo de IA ──encolar()──► queue.Queue(maxsize=N) ──► hilo "sqlite-escritor"
          (nunca espera)                                      agrupa y hace 1 commit/lote

    Decisiones de concurrencia:
      * `encolar()` usa `put_nowait`: si el disco se atasca (tarjeta SD lenta en
        la Jetson, checkpoint largo) la cola se llena y los eventos NUEVOS se
        descartan y contabilizan, en lugar de frenar el video. Preferimos perder
        un evento estadístico antes que congelar el escáner.
      * Un solo escritor por proceso: SQLite admite un único escritor a la vez;
        centralizarlo elimina la contención de candados dentro del proceso.
      * Vaciado por tamaño (`tam_lote`) o por tiempo (`intervalo_s`): baja
        latencia hasta el tablero (≤ intervalo_s) y alto rendimiento bajo carga.
      * Apagado ordenado con centinela + `atexit`: los eventos pendientes se
        escriben antes de cerrar el proceso.
    """

    _CENTINELA = object()

    def __init__(self, capacidad: int = 4096, tam_lote: int = 256, intervalo_s: float = 0.25) -> None:
        self._cola: queue.Queue[Any] = queue.Queue(maxsize=capacidad)
        self._tam_lote = tam_lote
        self._intervalo_s = intervalo_s
        self._lock = threading.Lock()          # protege solo los contadores
        self._escritos = 0
        self._descartados = 0
        self._lotes_fallidos = 0
        self._ultimo_error: str | None = None
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle, name="sqlite-escritor", daemon=True)
        self._hilo.start()

    # ---- Productores (cualquier hilo) --------------------------------------
    def encolar(self, evento: EventoClasificacion) -> bool:
        """No bloqueante. Devuelve False si el evento se descartó por cola llena."""
        if not self._activo:
            return False
        try:
            self._cola.put_nowait(evento)
            return True
        except queue.Full:
            with self._lock:
                self._descartados += 1
            return False

    def encolar_deteccion(self, material: str, contenedor: str, certidumbre: float) -> bool:
        return self.encolar(EventoClasificacion.crear(material, contenedor, certidumbre))

    # ---- Consumidor (hilo propio) ------------------------------------------
    def _bucle(self) -> None:
        lote: list[EventoClasificacion] = []
        limite = time.monotonic() + self._intervalo_s
        while True:
            espera = max(0.0, limite - time.monotonic())
            try:
                item = self._cola.get(timeout=espera if lote else 0.5)
            except queue.Empty:
                item = None
            if item is self._CENTINELA:
                self._vaciar(lote)
                return
            if item is not None:
                if not lote:
                    limite = time.monotonic() + self._intervalo_s
                lote.append(item)
            if lote and (len(lote) >= self._tam_lote or time.monotonic() >= limite):
                self._vaciar(lote)
                lote = []

    def _vaciar(self, lote: list[EventoClasificacion]) -> None:
        if not lote:
            return
        try:
            n = log_events(lote)
            with self._lock:
                self._escritos += n
        except Exception as exc:   # El hilo escritor jamás debe morir en silencio
            with self._lock:
                self._lotes_fallidos += 1
                self._descartados += len(lote)
                self._ultimo_error = f"{type(exc).__name__}: {exc}"
            _log.exception("No se pudo escribir un lote de %d eventos", len(lote))

    # ---- Ciclo de vida y observabilidad ------------------------------------
    def detener(self, timeout: float = 5.0) -> None:
        """Escribe lo pendiente y finaliza el hilo (idempotente)."""
        if not self._activo:
            return
        self._activo = False
        try:
            self._cola.put(self._CENTINELA, timeout=timeout)
        except queue.Full:
            _log.warning("Cola del registrador llena al apagar; se pierden eventos pendientes.")
            return
        self._hilo.join(timeout)

    @property
    def vivo(self) -> bool:
        return self._hilo.is_alive()

    def metricas(self) -> dict[str, Any]:
        with self._lock:
            return {"escritos": self._escritos, "descartados": self._descartados,
                    "pendientes": self._cola.qsize(), "lotes_fallidos": self._lotes_fallidos,
                    "ultimo_error": self._ultimo_error, "hilo_vivo": self.vivo}


_registrador: RegistradorAsincrono | None = None
_registrador_lock = threading.Lock()


def obtener_registrador() -> RegistradorAsincrono:
    """
    Singleton por proceso (double-checked locking). Todas las sesiones de
    Streamlit y todos los procesadores de video comparten el MISMO escritor.
    Si el hilo murió por algún motivo, se crea uno nuevo.
    """
    global _registrador
    reg = _registrador
    if reg is not None and reg.vivo:
        return reg
    with _registrador_lock:
        if _registrador is None or not _registrador.vivo:
            init_db()
            _registrador = RegistradorAsincrono()
            atexit.register(_registrador.detener)
        return _registrador
