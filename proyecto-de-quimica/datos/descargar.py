"""
datos/descargar.py — Descarga REAL de los bancos de imágenes públicos.

Deja cada fuente con la estructura exacta que espera `datos/construir_manifiesto.py`:

    <destino>/
      TACO/data/annotations.json + batch_*/...jpg         --taco
      trashnet/dataset-resized/<clase>/*.jpg              --trashnet
      zerowaste-f/.../{train,val,test}/labels.json + data --zerowaste
      off/productos_colombia.jsonl + off/imagenes/*.jpg   --off-productos / --off-imagenes
      LICENCIAS.csv

Uso (desde la carpeta del proyecto):
    python datos/descargar.py --todo                       # todo a ../datasets
    python datos/descargar.py --taco --trashnet            # solo algunas fuentes
    python datos/descargar.py --off --off-max-productos 3000
    python datos/descargar.py --zerowaste --borrar-zip     # ~7,5 GB de descarga

Tamaños aproximados:   TACO ~2,7 GB · TrashNet 43 MB · ZeroWaste-f 7,5 GB (zip) + ~8 GB
                       descomprimido · Open Food Facts (Colombia, 400 px) ~0,3–1 GB.

Es reanudable: si se corta, vuelva a ejecutar el mismo comando; los archivos completos se
omiten y las descargas grandes continúan donde quedaron (HTTP Range).

Solo usa la biblioteca estándar. Para descomprimir ZeroWaste (zip con contraseña AES) usa
7-Zip si está instalado o `pyzipper` (pip install pyzipper).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

AGENTE = "ProyectoQuimicaReciclaje/1.0 (proyecto academico; python-urllib)"

TACO_ANOTACIONES = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
TRASHNET_ZIP = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"
ZEROWASTE_REGISTRO = "6412647"   # Zenodo, versión 1.2.1
ZEROWASTE_ARCHIVO = "zerowaste-f-final.zip"
ZEROWASTE_URL = f"https://zenodo.org/records/{ZEROWASTE_REGISTRO}/files/{ZEROWASTE_ARCHIVO}?download=1"
ZEROWASTE_API = f"https://zenodo.org/api/records/{ZEROWASTE_REGISTRO}"
# Contraseña publicada por los autores en la página de Zenodo del dataset.
ZEROWASTE_CLAVE = "UP#1VuX409z4"
OFF_BUSQUEDA = "https://world.openfoodfacts.org/api/v2/search"
OFF_CAMPOS = ("code,countries_tags,packaging_tags,packaging_materials_tags,packaging_shapes_tags,"
              "categories_tags,image_front_url,image_packaging_url")
OFF_PAUSA_S = 6.5   # Límite oficial: 10 búsquedas/min por IP

LICENCIAS = [
    ("TACO", "CC BY 4.0", "https://github.com/pedropro/TACO"),
    ("TrashNet", "MIT", "https://github.com/garythung/trashnet"),
    ("ZeroWaste-f", "CC BY-NC 4.0 (solo uso NO comercial)", f"https://zenodo.org/records/{ZEROWASTE_REGISTRO}"),
    ("Open Food Facts", "Imágenes CC BY-SA · Base de datos ODbL", "https://world.openfoodfacts.org"),
]


# ===========================================================================
# Utilidades HTTP
# ===========================================================================
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _abrir(url: str, *, inicio: int = 0, timeout: float = 60.0):
    cab = {"User-Agent": AGENTE}
    if inicio:
        cab["Range"] = f"bytes={inicio}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=cab), timeout=timeout)


def leer_json(url: str, intentos: int = 5) -> dict | list:
    espera = 5.0
    for i in range(intentos):
        try:
            with _abrir(url) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if i == intentos - 1:
                raise
            log(f"  reintento {i + 1}/{intentos - 1} ({exc}); esperando {espera:.0f} s")
            time.sleep(espera)
            espera *= 2
    raise RuntimeError("inalcanzable")


def _humano(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def descargar_archivo(url: str, destino: Path, *, tam_esperado: int | None = None,
                      md5: str | None = None, intentos: int = 8, progreso: bool = True) -> Path:
    """Descarga reanudable (archivo .part + HTTP Range). Verifica tamaño/MD5 si se conocen."""
    if destino.exists() and (tam_esperado is None or destino.stat().st_size == tam_esperado):
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".part")
    for intento in range(intentos):
        ya = parcial.stat().st_size if parcial.exists() else 0
        try:
            with _abrir(url, inicio=ya, timeout=120) as r:
                if ya and r.status != 206:      # El servidor ignoró Range: empezar de cero
                    ya = 0
                total = ya + int(r.headers.get("Content-Length") or 0) or tam_esperado
                t0, ultimo = time.monotonic(), 0.0
                with parcial.open("ab" if ya else "wb") as f:
                    hechos = ya
                    while bloque := r.read(1 << 20):
                        f.write(bloque)
                        hechos += len(bloque)
                        ahora = time.monotonic()
                        if progreso and total and ahora - ultimo > 5:
                            vel = (hechos - ya) / max(ahora - t0, 1e-6)
                            log(f"  {destino.name}: {_humano(hechos)} / {_humano(total)} "
                                f"({100 * hechos / total:.1f} %, {_humano(vel)}/s)")
                            ultimo = ahora
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:   # Ya estaba completo
                break
            if intento == intentos - 1:
                raise
            log(f"  corte de red ({exc}); reanudando en 10 s…")
            time.sleep(10)
    if tam_esperado and parcial.stat().st_size != tam_esperado:
        raise IOError(f"{destino.name}: tamaño {parcial.stat().st_size} != {tam_esperado}")
    if md5:
        h = hashlib.md5()
        with parcial.open("rb") as f:
            while b := f.read(1 << 22):
                h.update(b)
        if h.hexdigest() != md5:
            parcial.unlink()
            raise IOError(f"{destino.name}: MD5 no coincide (archivo corrupto, se borró).")
    parcial.replace(destino)
    return destino


def _descargar_simple(url: str, destino: Path) -> bool:
    """Para miles de imágenes pequeñas: sin reanudación, True si quedó en disco."""
    if destino.exists() and destino.stat().st_size > 0:
        return True
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".part")
    for i in range(3):
        try:
            with _abrir(url, timeout=60) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(destino)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404, 410):     # Foto borrada de Flickr / OFF: no reintentar
                break
            time.sleep(2 * (i + 1))
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2 * (i + 1))
    tmp.unlink(missing_ok=True)
    return False


def espacio_libre_gb(ruta: Path) -> float:
    ruta.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(ruta).free / 1024 ** 3


# ===========================================================================
# TACO
# ===========================================================================
def descargar_taco(destino: Path, hilos: int) -> Path:
    raiz = destino / "TACO" / "data"
    raiz.mkdir(parents=True, exist_ok=True)
    ruta_json = descargar_archivo(TACO_ANOTACIONES, raiz / "annotations.json", progreso=False)
    datos = json.loads(ruta_json.read_text(encoding="utf-8"))
    tareas = []
    for im in datos["images"]:
        url = im.get("flickr_url") or im.get("flickr_640_url")
        if url:
            tareas.append((url, raiz / im["file_name"]))
    log(f"TACO: {len(tareas):,} imágenes, {hilos} descargas en paralelo → {raiz}")
    ok = falla = 0
    with ThreadPoolExecutor(hilos) as pool:
        futuros = {pool.submit(_descargar_simple, u, d): d for u, d in tareas}
        for k, fut in enumerate(as_completed(futuros), 1):
            ok += fut.result()
            falla += not fut.result()
            if k % 100 == 0 or k == len(futuros):
                log(f"  TACO {k:,}/{len(futuros):,}  (fallidas: {falla})")
    if falla:
        log(f"⚠️  TACO: {falla} fotos ya no existen en Flickr (normal: el manifiesto las omite).")
    return raiz


# ===========================================================================
# TrashNet
# ===========================================================================
def _extraer_zip(ruta_zip: Path, destino: Path, clave: str | None = None) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    siete = shutil.which("7z") or shutil.which("7za") or next(
        (p for p in (r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe")
         if Path(p).exists()), None)
    if clave and siete:
        log(f"  descomprimiendo con 7-Zip ({siete})…")
        subprocess.run([siete, "x", "-y", f"-p{clave}", f"-o{destino}", str(ruta_zip)], check=True,
                       stdout=subprocess.DEVNULL)
        return
    if clave:
        try:
            import pyzipper  # AES
            with pyzipper.AESZipFile(ruta_zip) as z:
                z.setpassword(clave.encode())
                _extraer_miembros(z, destino)
            return
        except ImportError:
            log("  (pyzipper no está instalado: se intenta con zipfile; si falla instale 7-Zip "
                "o ejecute: pip install pyzipper)")
    with zipfile.ZipFile(ruta_zip) as z:
        if clave:
            z.setpassword(clave.encode())
        _extraer_miembros(z, destino)


def _extraer_miembros(z, destino: Path) -> None:
    miembros = [m for m in z.namelist() if not m.startswith("__MACOSX/") and "/._" not in m]
    for k, m in enumerate(miembros, 1):
        z.extract(m, destino)
        if k % 2000 == 0:
            log(f"  extraídos {k:,}/{len(miembros):,}")


def descargar_trashnet(destino: Path) -> Path:
    carpeta = destino / "trashnet"
    final = carpeta / "dataset-resized"
    if final.is_dir() and any(final.glob("*/*.jpg")):
        log(f"TrashNet: ya existe {final}")
        return final
    ruta_zip = descargar_archivo(TRASHNET_ZIP, carpeta / "dataset-resized.zip")
    log("TrashNet: descomprimiendo…")
    _extraer_zip(ruta_zip, carpeta)
    n = len(list(final.glob("*/*.jpg")))
    log(f"TrashNet: {n:,} imágenes en {final}")
    return final


# ===========================================================================
# ZeroWaste-f
# ===========================================================================
def descargar_zerowaste(destino: Path, borrar_zip: bool) -> Path:
    carpeta = destino / "zerowaste-f"
    if any(carpeta.rglob("labels.json")):
        log(f"ZeroWaste: ya descomprimido en {carpeta}")
        return carpeta
    libre = espacio_libre_gb(destino)
    if libre < 17:
        sys.exit(f"❌ ZeroWaste necesita ~17 GB libres (zip + descomprimido); hay {libre:.1f} GB "
                 f"en {destino}. Use --destino en otro disco.")
    tam = md5 = None
    try:   # Tamaño y MD5 oficiales desde la API de Zenodo (si responde)
        reg = leer_json(ZEROWASTE_API, intentos=2)
        for f in reg.get("files", []):
            if f.get("key") == ZEROWASTE_ARCHIVO:
                tam = int(f["size"])
                md5 = str(f.get("checksum", "")).removeprefix("md5:") or None
    except Exception as exc:  # noqa: BLE001 - la verificación es opcional
        log(f"  (sin metadatos de Zenodo: {exc}; se descarga sin verificar MD5)")
    log(f"ZeroWaste-f: descargando {ZEROWASTE_ARCHIVO} ({_humano(tam) if tam else '≈7,5 GB'})…")
    ruta_zip = descargar_archivo(ZEROWASTE_URL, destino / ZEROWASTE_ARCHIVO, tam_esperado=tam, md5=md5)
    log("ZeroWaste-f: descomprimiendo (varios minutos)…")
    _extraer_zip(ruta_zip, carpeta, clave=ZEROWASTE_CLAVE)
    if not any(carpeta.rglob("labels.json")):
        sys.exit("❌ ZeroWaste se descomprimió pero no aparece ningún labels.json.")
    if borrar_zip:
        ruta_zip.unlink()
        log("  zip borrado (--borrar-zip)")
    return carpeta


# ===========================================================================
# Open Food Facts (Colombia)
# ===========================================================================
def _unir_tags(prod: dict) -> list[str]:
    vistos: dict[str, None] = {}
    for col in ("packaging_tags", "packaging_materials_tags", "packaging_shapes_tags"):
        for t in prod.get(col) or []:
            vistos.setdefault(str(t), None)
    return list(vistos)


def descargar_off(destino: Path, pais: str, max_productos: int | None, hilos: int) -> tuple[Path, Path]:
    from taxonomia import Taxonomia

    carpeta = destino / "off"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_productos = carpeta / f"productos_{pais}.jsonl"
    estado = carpeta / f".paginas_{pais}.json"
    hechas: set[int] = set(json.loads(estado.read_text())) if estado.exists() else set()

    # 1) Metadatos (paginado, respetando 10 búsquedas/min)
    params = {"countries_tags_en": pais, "fields": OFF_CAMPOS, "page_size": 100,
              "sort_by": "code"}
    primera = leer_json(f"{OFF_BUSQUEDA}?{urllib.parse.urlencode({**params, 'page': 1})}")
    total = int(primera.get("count", 0))
    paginas = (total - 1) // 100 + 1 if total else 0
    if max_productos:
        paginas = min(paginas, (max_productos - 1) // 100 + 1)
    log(f"Open Food Facts: {total:,} productos con país '{pais}'; {paginas} páginas "
        f"(≈{paginas * OFF_PAUSA_S / 60:.0f} min por el límite de la API).")
    with ruta_productos.open("a", encoding="utf-8") as salida:
        for pag in range(1, paginas + 1):
            if pag in hechas:
                continue
            datos = primera if pag == 1 else leer_json(
                f"{OFF_BUSQUEDA}?{urllib.parse.urlencode({**params, 'page': pag})}")
            for prod in datos.get("products", []):
                fila = {"code": str(prod.get("code", "")).strip(),
                        "countries_tags": prod.get("countries_tags") or [],
                        "packaging_tags": _unir_tags(prod),
                        "categories_tags": prod.get("categories_tags") or [],
                        "image_front_url": prod.get("image_front_url"),
                        "image_packaging_url": prod.get("image_packaging_url")}
                if fila["code"]:
                    salida.write(json.dumps(fila, ensure_ascii=False) + "\n")
            salida.flush()
            hechas.add(pag)
            estado.write_text(json.dumps(sorted(hechas)))
            log(f"  OFF página {pag}/{paginas}")
            if pag < paginas:
                time.sleep(OFF_PAUSA_S)

    # 2) Imágenes SOLO de los productos que la taxonomía sabe clasificar
    tax = Taxonomia.cargar()
    productos: dict[str, dict] = {}
    for linea in ruta_productos.read_text(encoding="utf-8").splitlines():
        p = json.loads(linea)
        productos[p["code"]] = p          # Sin duplicados si se reanudó
    dir_img = carpeta / "imagenes"
    tareas, por_clase = [], {}
    for codigo, p in productos.items():
        res = tax.resolver_off(p["packaging_tags"], p["categories_tags"])
        if res.estado != "mapeada":
            continue
        por_clase[res.clase] = por_clase.get(res.clase, 0) + 1
        for tipo in ("front", "packaging"):
            url = p.get(f"image_{tipo}_url")
            if url:
                tareas.append((url, dir_img / f"{codigo}_{tipo}.jpg"))
    log(f"OFF: {sum(por_clase.values()):,} de {len(productos):,} productos con clase mapeada "
        f"→ {len(tareas):,} imágenes (400 px).  Por clase: {dict(sorted(por_clase.items()))}")
    falla = 0
    with ThreadPoolExecutor(hilos) as pool:
        futuros = [pool.submit(_descargar_simple, u, d) for u, d in tareas]
        for k, fut in enumerate(as_completed(futuros), 1):
            falla += not fut.result()
            if k % 200 == 0 or k == len(futuros):
                log(f"  OFF imágenes {k:,}/{len(futuros):,} (fallidas: {falla})")
    return ruta_productos, dir_img


# ===========================================================================
def escribir_licencias(destino: Path) -> None:
    with (destino / "LICENCIAS.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fuente", "licencia", "url"])
        w.writerows(LICENCIAS)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Descarga TACO, TrashNet, ZeroWaste-f y Open Food Facts.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--destino", type=Path, default=RAIZ.parent / "datasets",
                   help="Carpeta FUERA del repo (por defecto ../datasets)")
    p.add_argument("--todo", action="store_true")
    p.add_argument("--taco", action="store_true")
    p.add_argument("--trashnet", action="store_true")
    p.add_argument("--zerowaste", action="store_true")
    p.add_argument("--off", action="store_true")
    p.add_argument("--off-pais", default="colombia")
    p.add_argument("--off-max-productos", type=int, default=None)
    p.add_argument("--hilos", type=int, default=8)
    p.add_argument("--borrar-zip", action="store_true", help="Borra el zip de ZeroWaste al terminar")
    args = p.parse_args(argv)
    if args.todo:
        args.taco = args.trashnet = args.zerowaste = args.off = True
    if not (args.taco or args.trashnet or args.zerowaste or args.off):
        p.error("Indique --todo o al menos una fuente (--taco --trashnet --zerowaste --off).")

    destino = args.destino.resolve()
    destino.mkdir(parents=True, exist_ok=True)
    log(f"Destino: {destino}  (libre: {espacio_libre_gb(destino):.1f} GB)")
    escribir_licencias(destino)
    rutas: dict[str, str] = {}
    if args.trashnet:
        rutas["--trashnet"] = str(descargar_trashnet(destino))
    if args.taco:
        rutas["--taco"] = str(descargar_taco(destino, args.hilos))
    if args.off:
        prod, img = descargar_off(destino, args.off_pais, args.off_max_productos, min(args.hilos, 4))
        rutas["--off-productos"], rutas["--off-imagenes"] = str(prod), str(img)
    if args.zerowaste:
        rutas["--zerowaste"] = str(descargar_zerowaste(destino, args.borrar_zip))

    (destino / "rutas.json").write_text(json.dumps(rutas, indent=2), encoding="utf-8")
    comando = " ".join(f'{k} "{v}"' for k, v in rutas.items())
    log("✔ Descarga terminada. Siguiente paso:")
    print(f"\n    python datos/construir_manifiesto.py {comando}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
