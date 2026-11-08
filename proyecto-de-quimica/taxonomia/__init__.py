"""
taxonomia — Fuente única de verdad de clases (Etapa A) y canecas (Etapa B)
==========================================================================
Carga `master.yaml` (16 clases que aprende la red) y `canecas.yaml` (reglas de la
Resolución 2184 de 2019), valida que sean coherentes y ofrece:

    tax = Taxonomia.cargar()                       # perfil nacional
    tax.resolver("taco", "Clear plastic bottle")   # -> Resolucion(clase='pet_botella', ...)
    tax.resolver("trashnet", "plastic")            # -> ambigua, candidatas=[pet, pead, otro]
    tax.caneca("plastico_flexible_bolsa")          # -> 'Blanca'
    tax.caneca("plastico_flexible_bolsa", sucio=True)   # -> 'Negra'
    tax.decidir({"papel": 0.52, "sanitario_y_rechazo": 0.41})  # -> incertidumbre -> 'Negra'

Ejecutar `python -m taxonomia` imprime un resumen y valida los archivos.

Autor: Johan Lasso
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError as _exc:  # pragma: no cover
    raise SystemExit("❌ Falta PyYAML:  pip install pyyaml") from _exc

DIR = Path(__file__).resolve().parent
RUTA_MASTER = DIR / "master.yaml"
RUTA_CANECAS = DIR / "canecas.yaml"
CANECAS_VALIDAS = {"Blanca", "Negra", "Verde", "Ninguna", None}
FUENTES = ("taco", "trashnet", "zerowaste", "off", "propias")


def normalizar(texto: str) -> str:
    """
    snake_case ASCII, sin prefijo de idioma de Open Food Facts.

    >>> normalizar("Clear plastic bottle")
    'clear_plastic_bottle'
    >>> normalizar("en:pet-1-polyethylene-terephthalate")
    'pet_1_polyethylene_terephthalate'
    >>> normalizar("Cartón  Ondulado")
    'carton_ondulado'
    """
    texto = re.sub(r"^[a-z]{2}:", "", str(texto).strip().lower())
    ascii_ = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_).strip("_")


@dataclass(frozen=True)
class Clase:
    id: int
    nombre: str
    descripcion: str


@dataclass(frozen=True)
class Resolucion:
    """Resultado de traducir una etiqueta de una fuente a la taxonomía maestra."""

    estado: str                        # "mapeada" | "ambigua" | "sin_mapeo"
    clase: str | None = None
    candidatas: tuple[str, ...] = ()

    @property
    def usable_clasificador(self) -> bool:
        return self.estado == "mapeada"


@dataclass(frozen=True)
class Decision:
    """Salida de la Etapa B para la UI."""

    clase: str | None
    caneca: str | None
    confianza: float
    incierto: bool
    nota: str = ""
    preguntar_limpieza: bool = False


@dataclass
class Taxonomia:
    clases: list[Clase]
    reglas: dict[str, dict[str, Any]]
    incertidumbre: dict[str, Any]
    _alias: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)
    _ambiguas: dict[tuple[str, str], tuple[str, ...]] = field(default_factory=dict, repr=False)
    reglas_off: list[dict[str, Any]] = field(default_factory=list, repr=False)
    perfil: str = "nacional"

    # ------------------------------------------------------------------ carga
    @classmethod
    def cargar(cls, ruta_master: Path = RUTA_MASTER, ruta_canecas: Path = RUTA_CANECAS,
               perfil: str | None = None) -> "Taxonomia":
        master = yaml.safe_load(Path(ruta_master).read_text(encoding="utf-8"))
        canecas = yaml.safe_load(Path(ruta_canecas).read_text(encoding="utf-8"))

        clases = [Clase(int(c["id"]), c["nombre"], " ".join(str(c.get("descripcion", "")).split()))
                  for c in master["clases"]]
        alias: dict[tuple[str, str], str] = {}
        for c in master["clases"]:
            for fuente, etiquetas in (c.get("alias") or {}).items():
                for e in etiquetas or []:
                    clave = (fuente, normalizar(e))
                    if clave in alias and alias[clave] != c["nombre"]:
                        raise ValueError(f"Alias duplicado {clave}: {alias[clave]} y {c['nombre']}")
                    alias[clave] = c["nombre"]
        ambiguas = {(fuente, normalizar(e)): tuple(cands or ())
                    for fuente, tabla in (master.get("ambiguas") or {}).items()
                    for e, cands in (tabla or {}).items()}

        reglas = {k: dict(v or {}) for k, v in canecas["reglas"].items()}
        perfil = perfil or canecas.get("perfil", "nacional")
        if perfil != canecas.get("perfil", "nacional"):
            extra = (canecas.get("perfiles") or {}).get(perfil)
            if extra is None:
                raise ValueError(f"Perfil desconocido: {perfil!r}")
            for k, v in extra.items():
                reglas[k] = {**reglas.get(k, {}), **v}

        tax = cls(clases, reglas, dict(canecas.get("incertidumbre") or {}), alias, ambiguas,
                  list(master.get("reglas_off") or []), perfil)
        tax.validar()
        return tax

    def validar(self) -> None:
        """Errores de coherencia entre master.yaml y canecas.yaml."""
        ids = [c.id for c in self.clases]
        if ids != list(range(len(ids))):
            raise ValueError(f"Los ids deben ser 0..N-1 en orden; hay {ids}")
        nombres = set(self.nombres)
        if len(nombres) != len(self.clases):
            raise ValueError("Nombres de clase repetidos en master.yaml")
        if set(self.reglas) != nombres:
            faltan, sobran = nombres - set(self.reglas), set(self.reglas) - nombres
            raise ValueError(f"canecas.yaml no coincide con master.yaml. Faltan: {sorted(faltan)}; "
                             f"sobran: {sorted(sobran)}")
        for nombre, r in self.reglas.items():
            for campo in ("caneca", "si_sucio"):
                if campo in r and r[campo] not in CANECAS_VALIDAS:
                    raise ValueError(f"{nombre}.{campo} = {r[campo]!r} no es una caneca válida")
        for (fuente, _), cands in self._ambiguas.items():
            if fuente not in FUENTES or not set(cands) <= nombres:
                raise ValueError(f"Ambigua inválida en {fuente}: {cands}")
        for regla in self.reglas_off:
            if regla["clase"] not in nombres:
                raise ValueError(f"reglas_off apunta a una clase inexistente: {regla['clase']}")

    # -------------------------------------------------------------- consultas
    @property
    def nombres(self) -> list[str]:
        return [c.nombre for c in self.clases]

    def id_de(self, nombre: str) -> int:
        return self.nombres.index(nombre)

    def resolver(self, fuente: str, etiqueta: str, supercategoria: str | None = None) -> Resolucion:
        """Traduce la etiqueta de una fuente. La súper-categoría solo se usa como respaldo."""
        for e in (etiqueta, supercategoria):
            if not e:
                continue
            clave = (fuente, normalizar(e))
            if clave in self._alias:
                return Resolucion("mapeada", self._alias[clave])
            if clave in self._ambiguas:
                return Resolucion("ambigua", None, self._ambiguas[clave])
            # Fotos propias: una carpeta con el nombre oficial de la clase (pet_botella/,
            # metal_lata/...) siempre vale, además de los alias amigables (bolsa_blanca/, pet/...).
            if fuente == "propias" and normalizar(e) in self.nombres:
                return Resolucion("mapeada", normalizar(e))
        return Resolucion("sin_mapeo")

    def resolver_off(self, packaging_tags: list[str] | None,
                     categories_tags: list[str] | None = None) -> Resolucion:
        """Clase de un producto de Open Food Facts a partir de sus etiquetas."""
        etiquetas = {normalizar(t) for t in (packaging_tags or [])}
        etiquetas |= {normalizar(t) for t in (categories_tags or [])}
        if not etiquetas:
            return Resolucion("sin_mapeo")
        for regla in self.reglas_off:
            todas = {normalizar(t) for t in regla.get("todas", [])}
            alguna = {normalizar(t) for t in regla.get("alguna", [])}
            if todas <= etiquetas and (not alguna or alguna & etiquetas):
                return Resolucion("mapeada", regla["clase"])
        return Resolucion("sin_mapeo")

    def caneca(self, clase: str, sucio: bool | None = None) -> str | None:
        """Etapa B. `sucio=None` = desconocido -> se usa la caneca principal."""
        r = self.reglas[clase]
        if sucio and "si_sucio" in r:
            return r["si_sucio"]
        return r.get("caneca")

    def decidir(self, probabilidades: Mapping[str, float], sucio: bool | None = None) -> Decision:
        """
        Aplica la política de incertidumbre y luego la regla de caneca.

        >>> t = Taxonomia.cargar()
        >>> t.decidir({"papel": 0.52, "sanitario_y_rechazo": 0.41}).caneca
        'Negra'
        >>> t.decidir({"vidrio": 0.93, "pet_botella": 0.04}).caneca
        'Blanca'
        """
        orden = sorted(probabilidades.items(), key=lambda kv: kv[1], reverse=True)
        if not orden:
            return Decision(None, self.incertidumbre.get("caneca_por_defecto", "Negra"), 0.0, True)
        clase, p1 = orden[0]
        p2 = orden[1][1] if len(orden) > 1 else 0.0
        umbral = float(self.incertidumbre.get("umbral_confianza", 0.6))
        margen = float(self.incertidumbre.get("margen_top2", 0.15))
        if p1 < umbral or (p1 - p2) < margen:
            return Decision(clase, self.incertidumbre.get("caneca_por_defecto", "Negra"), p1, True,
                            "Incertidumbre: por precaución, caneca negra.")
        r = self.reglas[clase]
        return Decision(clase, self.caneca(clase, sucio), p1, False, r.get("nota", ""),
                        bool(r.get("preguntar_limpieza", False)))


def _resumen() -> None:  # pragma: no cover - utilidad de consola
    t = Taxonomia.cargar()
    print(f"Taxonomía v1 · {len(t.clases)} clases · perfil {t.perfil}")
    for c in t.clases:
        print(f"  {c.id:2d} {c.nombre:26s} -> {str(t.caneca(c.nombre)):8s} "
              f"(sucio: {t.caneca(c.nombre, sucio=True)})")
    print(f"Alias: {len(t._alias)} · Ambiguas: {len(t._ambiguas)} · Reglas OFF: {len(t.reglas_off)}")


if __name__ == "__main__":  # pragma: no cover
    _resumen()
