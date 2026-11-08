"""
construir_manifiesto.py — Manifiesto común de datasets heterogéneos (offline)
=============================================================================
Lee las fuentes YA DESCARGADAS en disco y produce UNA tabla con una fila por
objeto (o por imagen, si la fuente no trae cajas), etiquetada con la Taxonomía
Maestra (`taxonomia/master.yaml`). El pipeline de entrenamiento solo consume este
manifiesto: nunca vuelve a leer los formatos originales.

Fuentes soportadas:
    --taco       <TACO/data>          annotations.json + batch_*/ (COCO, bbox + polígonos)
    --zerowaste  <zerowaste-f>        cualquier */labels.json o *.json COCO (train/val/test)
    --trashnet   <dataset-resized>    carpetas cardboard/ glass/ metal/ paper/ plastic/ trash/
    --off-productos <archivo>         export de Open Food Facts (.parquet / .jsonl / .csv)
    --off-imagenes  <carpeta>         imágenes ya descargadas: <codigo>_*.jpg o <codigo>/*.jpg
    --propias    <fotos_propias>      <sesion>/<clase>/*.jpg  o  <clase>/*.jpg

Columnas del manifiesto:
    id_objeto              Identificador único y estable ("taco:batch_1/000006.jpg#17")
    fuente                 taco | zerowaste | trashnet | off | propias
    ruta                   Ruta absoluta de la imagen
    ancho, alto            Tamaño declarado (COCO) o leído del encabezado
    bbox_x/y/w/h           Caja COCO en píxeles; vacía = imagen completa
    tiene_mascara          True si hay polígono/RLE (útil para copy-paste)
    etiqueta_origen        Categoría original de la fuente
    supercategoria_origen  Súper-categoría (si existe)
    estado_mapeo           mapeada | ambigua | sin_mapeo
    clase_master, clase_id Clase de la taxonomía (vacía / -1 si no está mapeada)
    candidatas             Clases posibles si es ambigua ("pet_botella|pead_pp_envase")
    uso_detector           True si aporta una caja al detector "residuo"
    uso_clasificador       True si aporta un recorte etiquetado al clasificador
    split                  entrenamiento | validacion | prueba
    grupo_split            Unidad que nunca se parte entre conjuntos (lote, producto, sesión)
    licencia               Licencia de la fuente

Particiones (sin fugas):
    * TACO       por lote (batch_*): --taco-val / --taco-prueba (por defecto 12,13 / 14,15,
                 los mismos lotes de prueba que entrenamiento/reentrenar.py).
    * ZeroWaste  por la carpeta oficial (train / val / test).
    * TrashNet   70/15/15 estratificado con semilla 42 (igual que el entrenamiento original).
    * OFF        por código de barras, 85 % entrenamiento / 15 % validación. NUNCA prueba:
                 son fotos de producto nuevo, no de residuos.
    * Propias    por sesión de toma; las sesiones cuyo nombre empieza por "prueba" van a
                 prueba. Esta es la prueba que más importa: tu cámara, tu luz.

Uso:
    python datos/construir_manifiesto.py --taco ../TACO/data --trashnet entrenamiento/dataset-resized \\
        --propias fotos_propias --salida data/manifiesto.parquet
    python datos/construir_manifiesto.py --taco ../TACO/data --zerowaste ../zerowaste-f \\
        --off-productos ../off/food.parquet --off-imagenes ../off/imagenes

Autor: Johan Lasso
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402

from taxonomia import Taxonomia, normalizar  # noqa: E402

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LICENCIAS = {
    "taco": "CC BY 4.0",
    "zerowaste": "CC BY-NC 4.0 (no comercial)",
    "trashnet": "MIT",
    "off": "CC BY-SA 3.0/4.0 (imágenes) · ODbL (base de datos)",
    "propias": "Propia",
}
CARPETAS_TRASHNET = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
SPLITS = ("entrenamiento", "validacion", "prueba")
_SPLIT_CARPETA = {"train": "entrenamiento", "training": "entrenamiento", "val": "validacion",
                  "valid": "validacion", "validation": "validacion", "test": "prueba"}


@dataclass
class Fila:
    id_objeto: str
    fuente: str
    ruta: str
    ancho: int | None
    alto: int | None
    bbox_x: float | None
    bbox_y: float | None
    bbox_w: float | None
    bbox_h: float | None
    tiene_mascara: bool
    etiqueta_origen: str
    supercategoria_origen: str | None
    estado_mapeo: str
    clase_master: str | None
    clase_id: int
    candidatas: str
    uso_detector: bool
    uso_clasificador: bool
    split: str
    grupo_split: str
    licencia: str


# ===========================================================================
# Utilidades
# ===========================================================================
def particion_hash(grupo: str, p_val: float, p_prueba: float) -> str:
    """
    Conjunto estable a partir de un identificador de grupo (independiente del orden).

    >>> particion_hash("7702004003508", 0.15, 0.0) in ("entrenamiento", "validacion")
    True
    """
    u = (zlib.crc32(grupo.encode()) & 0xFFFFFFFF) / 2**32
    if u < p_prueba:
        return "prueba"
    if u < p_prueba + p_val:
        return "validacion"
    return "entrenamiento"


def tam_imagen(ruta: Path) -> tuple[int | None, int | None]:
    """(ancho, alto) leyendo solo el encabezado, ya con la orientación EXIF aplicada."""
    try:
        from PIL import Image
        with Image.open(ruta) as im:
            ancho, alto = im.size
            orientacion = im.getexif().get(0x0112, 1)
        return (alto, ancho) if orientacion in (5, 6, 7, 8) else (ancho, alto)
    except Exception:
        return None, None


def imagenes_en(carpeta: Path) -> list[Path]:
    return sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES)


def _fila(tax: Taxonomia, *, fuente: str, id_objeto: str, ruta: Path, etiqueta: str,
          supercategoria: str | None, split: str, grupo: str, ancho=None, alto=None,
          bbox=None, mascara=False, resolucion=None) -> Fila:
    res = resolucion or tax.resolver(fuente, etiqueta, supercategoria)
    x, y, w, h = bbox if bbox else (None, None, None, None)
    return Fila(
        id_objeto=id_objeto, fuente=fuente, ruta=ruta.resolve().as_posix(),
        ancho=ancho, alto=alto, bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h, tiene_mascara=mascara,
        etiqueta_origen=etiqueta, supercategoria_origen=supercategoria,
        estado_mapeo=res.estado, clase_master=res.clase,
        clase_id=tax.id_de(res.clase) if res.clase else -1,
        candidatas="|".join(res.candidatas),
        uso_detector=bbox is not None,
        uso_clasificador=res.usable_clasificador,
        split=split, grupo_split=grupo, licencia=LICENCIAS[fuente],
    )


# ===========================================================================
# Lectores por fuente
# ===========================================================================
def _leer_coco(ruta_json: Path) -> dict:
    datos = json.loads(ruta_json.read_text(encoding="utf-8"))
    if not {"images", "annotations", "categories"} <= set(datos):
        raise ValueError(f"{ruta_json} no es un archivo COCO")
    return datos


def _resolver_imagen(base: Path, nombre: str) -> Path | None:
    for candidato in (base / nombre, base / "data" / nombre, base / "images" / nombre):
        if candidato.is_file():
            return candidato
    return None


def filas_coco(tax: Taxonomia, ruta_json: Path, *, fuente: str, raiz: Path,
               split_de_imagen, lado_minimo: int = 8) -> Iterator[Fila]:
    """Una fila por anotación válida de un archivo COCO (TACO, ZeroWaste, Roboflow)."""
    datos = _leer_coco(ruta_json)
    base = ruta_json.parent
    cats = {c["id"]: (c["name"], c.get("supercategory")) for c in datos["categories"]}
    imgs = {im["id"]: im for im in datos["images"]}
    faltantes = 0
    rutas: dict[int, Path | None] = {}
    for an in datos["annotations"]:
        if an.get("iscrowd") or len(an.get("bbox") or ()) != 4:
            continue
        x, y, w, h = (float(v) for v in an["bbox"])
        if min(w, h) < lado_minimo:
            continue
        meta = imgs.get(an["image_id"])
        if meta is None:
            continue
        if an["image_id"] not in rutas:
            rutas[an["image_id"]] = _resolver_imagen(base, meta["file_name"])
            faltantes += rutas[an["image_id"]] is None
        ruta = rutas[an["image_id"]]
        if ruta is None:
            continue
        nombre, superc = cats.get(an["category_id"], (str(an["category_id"]), None))
        relativo = ruta.relative_to(raiz).as_posix() if ruta.is_relative_to(raiz) else ruta.name
        split, grupo = split_de_imagen(relativo, ruta)
        yield _fila(tax, fuente=fuente, id_objeto=f"{fuente}:{relativo}#{an['id']}", ruta=ruta,
                    etiqueta=nombre, supercategoria=superc, split=split, grupo=grupo,
                    ancho=meta.get("width"), alto=meta.get("height"), bbox=(x, y, w, h),
                    mascara=bool(an.get("segmentation")))
    if faltantes:
        print(f"⚠️  {fuente}: {faltantes} imágenes de {ruta_json.name} no están en disco.")


def filas_taco(tax: Taxonomia, raiz: Path, lotes_val: set[str], lotes_prueba: set[str]) -> Iterator[Fila]:
    raiz = raiz.resolve()
    ruta_json = raiz / "annotations.json"
    if not ruta_json.is_file():
        sys.exit(f"❌ No existe {ruta_json} (¿ejecutaste download.py de TACO?).")

    def split(relativo: str, _ruta: Path) -> tuple[str, str]:
        lote = relativo.split("/")[0]
        if lote in lotes_prueba:
            return "prueba", lote
        return ("validacion" if lote in lotes_val else "entrenamiento"), lote

    yield from filas_coco(tax, ruta_json, fuente="taco", raiz=raiz, split_de_imagen=split)


def filas_zerowaste(tax: Taxonomia, raiz: Path) -> Iterator[Fila]:
    raiz = raiz.resolve()
    jsons = sorted(p for p in raiz.rglob("*.json") if p.stat().st_size > 0)
    usados = 0
    for ruta_json in jsons:
        try:
            _leer_coco(ruta_json)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        usados += 1
        partes = [p.lower() for p in ruta_json.relative_to(raiz).parts]
        oficial = next((_SPLIT_CARPETA[p] for p in partes if p in _SPLIT_CARPETA), None)

        def split(relativo: str, _ruta: Path, oficial=oficial) -> tuple[str, str]:
            # Fotogramas de video: agrupar por carpeta evita fugas entre cuadros vecinos.
            grupo = "/".join(relativo.split("/")[:-1]) or relativo
            return (oficial or particion_hash(grupo, 0.15, 0.15)), grupo

        yield from filas_coco(tax, ruta_json, fuente="zerowaste", raiz=raiz, split_de_imagen=split)
    if not usados:
        sys.exit(f"❌ No se encontró ningún JSON COCO dentro de {raiz}.")


def filas_trashnet(tax: Taxonomia, raiz: Path) -> Iterator[Fila]:
    raiz = raiz.resolve()
    if not any((raiz / c).is_dir() for c in CARPETAS_TRASHNET) and (raiz / "dataset-resized").is_dir():
        raiz = raiz / "dataset-resized"
    rutas, y = [], []
    for k, c in enumerate(CARPETAS_TRASHNET):
        archivos = sorted((raiz / c).glob("*.jpg"))
        rutas += archivos
        y += [k] * len(archivos)
    if not rutas:
        sys.exit(f"❌ {raiz} no tiene la estructura de TrashNet.")
    try:  # Misma partición que entrenar_trashnet.py / reentrenar.py
        import numpy as np
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(rutas))
        i_ent, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
        i_val, i_pru = train_test_split(i_tmp, test_size=0.50, stratify=np.array(y)[i_tmp],
                                        random_state=42)
        conjunto = {**{int(i): "entrenamiento" for i in i_ent},
                    **{int(i): "validacion" for i in i_val}, **{int(i): "prueba" for i in i_pru}}
    except ImportError:
        print("⚠️  scikit-learn no está instalado: TrashNet se parte por hash (no comparable).")
        conjunto = {i: particion_hash(r.name, 0.15, 0.15) for i, r in enumerate(rutas)}
    for i, ruta in enumerate(rutas):
        ancho, alto = tam_imagen(ruta)
        yield _fila(tax, fuente="trashnet", id_objeto=f"trashnet:{ruta.parent.name}/{ruta.name}",
                    ruta=ruta, etiqueta=ruta.parent.name, supercategoria=None,
                    split=conjunto[i], grupo=ruta.name, ancho=ancho, alto=alto)


def _leer_productos_off(ruta: Path) -> pd.DataFrame:
    if ruta.suffix == ".parquet":
        return pd.read_parquet(ruta)
    if ruta.suffix in (".jsonl", ".json") or ruta.name.endswith(".jsonl.gz"):
        return pd.read_json(ruta, lines=True)
    # El CSV oficial de Open Food Facts viene separado por tabuladores.
    with ruta.open(encoding="utf-8") as f:
        separador = "\t" if "\t" in f.readline() else ","
    return pd.read_csv(ruta, sep=separador, low_memory=False)


def _como_lista(valor) -> list[str]:
    """Las columnas *_tags llegan como lista, arreglo, o texto 'en:a,en:b'."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return []
    if isinstance(valor, str):
        return [v for v in valor.split(",") if v]
    try:
        return [str(v) for v in valor]
    except TypeError:
        return []


def filas_off(tax: Taxonomia, ruta_productos: Path, dir_imagenes: Path,
              pais: str = "colombia", max_por_producto: int = 2) -> Iterator[Fila]:
    """
    Productos de Open Food Facts vendidos en `pais`, con imágenes YA descargadas.

    No se usa la red: las imágenes deben estar en `dir_imagenes` como
    <codigo>_<n>.jpg, <codigo>.jpg o <codigo>/<n>.jpg (ver datos/README del pipeline).
    """
    df = _leer_productos_off(ruta_productos)
    for col in ("code", "countries_tags", "packaging_tags"):
        if col not in df.columns:
            sys.exit(f"❌ El export de Open Food Facts no tiene la columna {col!r}.")
    pais = normalizar(pais)
    df = df[df["countries_tags"].map(lambda v: pais in {normalizar(t) for t in _como_lista(v)})]
    print(f"ℹ️  Open Food Facts: {len(df):,} productos con país '{pais}'.")
    dir_imagenes = dir_imagenes.resolve()
    indice: dict[str, list[Path]] = {}
    for p in imagenes_en(dir_imagenes):
        codigo = p.parent.name if p.parent != dir_imagenes else p.stem.split("_")[0]
        indice.setdefault(codigo, []).append(p)
    sin_imagen = 0
    for _, prod in df.iterrows():
        codigo = str(prod["code"]).strip()
        res = tax.resolver_off(_como_lista(prod.get("packaging_tags")),
                               _como_lista(prod.get("categories_tags")))
        if res.estado != "mapeada":
            continue
        imagenes = indice.get(codigo, [])[:max_por_producto]
        sin_imagen += not imagenes
        split = particion_hash(codigo, 0.15, 0.0)
        for ruta in imagenes:
            ancho, alto = tam_imagen(ruta)
            yield _fila(tax, fuente="off", id_objeto=f"off:{codigo}/{ruta.name}", ruta=ruta,
                        etiqueta=",".join(_como_lista(prod.get("packaging_tags")))[:200],
                        supercategoria=None, split=split, grupo=codigo,
                        ancho=ancho, alto=alto, resolucion=res)
    if sin_imagen:
        print(f"ℹ️  Open Food Facts: {sin_imagen:,} productos mapeados sin imagen en {dir_imagenes}.")


def filas_propias(tax: Taxonomia, raiz: Path) -> Iterator[Fila]:
    """<raiz>/<sesion>/<clase>/*.jpg  o  <raiz>/<clase>/*.jpg (sin sesiones)."""
    raiz = raiz.resolve()
    for ruta in imagenes_en(raiz):
        partes = ruta.relative_to(raiz).parts
        if len(partes) >= 3:
            sesion, etiqueta = partes[0], partes[-2]
            split = "prueba" if normalizar(sesion).startswith("prueba") else \
                particion_hash(sesion, 0.25, 0.0)
            grupo = sesion
        else:
            etiqueta = partes[-2] if len(partes) == 2 else "sin_clase"
            grupo = ruta.name
            split = particion_hash(grupo, 0.15, 0.15)
        ancho, alto = tam_imagen(ruta)
        yield _fila(tax, fuente="propias", id_objeto=f"propias:{'/'.join(partes)}", ruta=ruta,
                    etiqueta=etiqueta, supercategoria=None, split=split, grupo=grupo,
                    ancho=ancho, alto=alto)


# ===========================================================================
# Ensamblado, validación y reporte
# ===========================================================================
def construir(tax: Taxonomia, generadores: Iterable[Iterator[Fila]]) -> pd.DataFrame:
    filas = [asdict(f) for g in generadores for f in g]
    df = pd.DataFrame(filas, columns=list(Fila.__dataclass_fields__))
    if df.empty:
        return df
    duplicados = df["id_objeto"].duplicated()
    if duplicados.any():
        print(f"⚠️  {int(duplicados.sum())} id_objeto duplicados: se conserva el primero.")
        df = df[~duplicados]
    # Un mismo grupo no puede quedar en dos conjuntos (fuga entre entrenamiento y prueba).
    fugas = df.groupby(["fuente", "grupo_split"])["split"].nunique()
    if (fugas > 1).any():
        raise RuntimeError(f"Grupos partidos entre conjuntos: {list(fugas[fugas > 1].index)[:5]}")
    return df.reset_index(drop=True)


def reporte(df: pd.DataFrame) -> str:
    if df.empty:
        return "Manifiesto vacío."
    lineas = [f"Manifiesto: {len(df):,} filas · {df['ruta'].nunique():,} imágenes",
              f"  Para el detector (cajas):         {int(df['uso_detector'].sum()):,}",
              f"  Para el clasificador (etiquetas): {int(df['uso_clasificador'].sum()):,}", ""]
    clf = df[df["uso_clasificador"]]
    if not clf.empty:
        tabla = pd.crosstab([clf["clase_id"], clf["clase_master"]], clf["split"], margins=True)
        tabla = tabla.reindex(columns=[c for c in (*SPLITS, "All") if c in tabla.columns])
        lineas += ["Clasificador · objetos por clase y conjunto:", tabla.to_string(), ""]
        lineas += ["Clasificador · objetos por clase y fuente:",
                   pd.crosstab(clf["clase_master"], clf["fuente"]).to_string(), ""]
    sin = df[df["estado_mapeo"] == "sin_mapeo"]
    if not sin.empty:
        top = Counter(zip(sin["fuente"], sin["etiqueta_origen"])).most_common(20)
        lineas += ["Etiquetas SIN mapeo (agrégalas a taxonomia/master.yaml):"]
        lineas += [f"  {f:10s} {e[:60]:60s} {n:>6}" for (f, e), n in top]
    amb = df[df["estado_mapeo"] == "ambigua"]
    if not amb.empty:
        lineas += ["", "Etiquetas AMBIGUAS (no entran al clasificador por defecto):"]
        lineas += [f"  {f:10s} {e:30s} {n:>6}" for (f, e), n in
                   Counter(zip(amb["fuente"], amb["etiqueta_origen"])).most_common(10)]
    return "\n".join(lineas)


def guardar(df: pd.DataFrame, salida: Path) -> Path:
    salida.parent.mkdir(parents=True, exist_ok=True)
    if salida.suffix == ".parquet":
        try:
            df.to_parquet(salida, index=False)
            return salida
        except ImportError:
            salida = salida.with_suffix(".csv")
            print("ℹ️  pyarrow no está instalado: se guarda en CSV.")
    df.to_csv(salida, index=False, encoding="utf-8")
    return salida


def rebasear_rutas_si_es_necesario(df: pd.DataFrame, raiz: Path = RAIZ) -> pd.DataFrame:
    """Si el repositorio fue movido a otra ruta o letra de unidad, rebasea las rutas al directorio actual."""
    if df.empty or "ruta" not in df.columns:
        return df
    muestra = df["ruta"].dropna().head(10)
    if not muestra.empty and all(Path(p).exists() for p in muestra):
        return df

    def _arreglar(r: str) -> str:
        if not r or Path(r).exists():
            return r
        partes = Path(r).parts
        if "proyecto-de-quimica" in partes:
            idx = partes.index("proyecto-de-quimica")
            cand = raiz / Path(*partes[idx + 1:])
            if cand.exists():
                return cand.resolve().as_posix()
        for i in range(len(partes) - 1):
            cand = raiz / Path(*partes[i:])
            if cand.exists():
                return cand.resolve().as_posix()
        return r

    df["ruta"] = df["ruta"].apply(_arreglar)
    return df


def cargar_manifiesto(ruta: Path) -> pd.DataFrame:
    ruta = Path(ruta)
    if not ruta.exists() and ruta.suffix == ".parquet" and ruta.with_suffix(".csv").exists():
        ruta = ruta.with_suffix(".csv")
    df = pd.read_parquet(ruta) if ruta.suffix == ".parquet" else pd.read_csv(ruta)
    return rebasear_rutas_si_es_necesario(df)


def _lotes(texto: str) -> set[str]:
    return {f"batch_{t.strip().removeprefix('batch_')}" for t in texto.split(",") if t.strip()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--taco", type=Path)
    p.add_argument("--taco-val", default="12,13", help="Lotes TACO de validación")
    p.add_argument("--taco-prueba", default="14,15", help="Lotes TACO de prueba")
    p.add_argument("--zerowaste", type=Path)
    p.add_argument("--trashnet", type=Path)
    p.add_argument("--off-productos", type=Path)
    p.add_argument("--off-imagenes", type=Path)
    p.add_argument("--off-pais", default="colombia")
    p.add_argument("--propias", type=Path)
    p.add_argument("--perfil", default=None, help="Perfil de canecas.yaml")
    p.add_argument("--salida", type=Path, default=RAIZ / "data" / "manifiesto.parquet")
    args = p.parse_args(argv)

    tax = Taxonomia.cargar(perfil=args.perfil)
    generadores = []
    if args.taco:
        generadores.append(filas_taco(tax, args.taco, _lotes(args.taco_val), _lotes(args.taco_prueba)))
    if args.zerowaste:
        generadores.append(filas_zerowaste(tax, args.zerowaste))
    if args.trashnet:
        generadores.append(filas_trashnet(tax, args.trashnet))
    if args.off_productos:
        if not args.off_imagenes:
            p.error("--off-productos requiere --off-imagenes")
        generadores.append(filas_off(tax, args.off_productos, args.off_imagenes, args.off_pais))
    if args.propias:
        generadores.append(filas_propias(tax, args.propias))
    if not generadores:
        p.error("Indica al menos una fuente (--taco, --zerowaste, --trashnet, --off-productos, --propias).")

    df = construir(tax, generadores)
    ruta = guardar(df, args.salida)
    texto = reporte(df)
    ruta.with_name(ruta.stem + "_reporte.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\n✅ Manifiesto guardado en {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
