"""Pruebas de extremo a extremo de ingesta_publica.py sobre un COCO sintético."""

import sqlite3

import database as db
import ingesta_publica as ip


def _contar(origen=None):
    with db.conectar() as con:
        if origen:
            return con.execute("SELECT COUNT(*) FROM clasificaciones WHERE origen = ?",
                               (origen,)).fetchone()[0]
        return con.execute("SELECT COUNT(*) FROM clasificaciones").fetchone()[0]


def _ingestar(ruta, *extra):
    return ip.main(["--ruta", str(ruta), "--aumentos", "1", "--guardar-todo",
                    "--motor", "Motor Geométrico (Forma/Contornos)", *extra])


def test_lectura_coco(coco_sintetico):
    muestras, sin_mapeo = ip.muestras_coco(ip.buscar_anotaciones(coco_sintetico), coco_sintetico,
                                           lado_minimo=48, max_por_imagen=3,
                                           incluir_sin_mapeo=False)
    assert len(muestras) == 7                       # 8 objetos - 1 sin mapeo
    assert sin_mapeo == {"Categoria inventada": 1}
    assert all(m.bbox[2] >= 48 for m in muestras)   # La caja diminuta se descartó
    criticas = {m.detalle for m in muestras if m.critica}
    assert criticas == {"Crisp packet", "Foam food container", "Aerosol"}


def test_orientacion_exif(coco_sintetico):
    """La foto guardada «acostada» con EXIF se lee con el tamaño del JSON (640x480)."""
    bgr = ip.leer_imagen(coco_sintetico / "batch_1" / "000.jpg", (640, 480))
    assert bgr.shape[:2] == (480, 640)


def test_ingesta_inserta_en_wal_con_clases_finas(coco_sintetico):
    assert _ingestar(coco_sintetico) == 0
    # 7 objetos x 2 variantes (original + 1 aumento)
    assert _contar() == 14
    assert _contar("taco") == 7 and _contar("taco_aug") == 7
    con = sqlite3.connect(db.RUTA_BD)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    etiquetas = {r[0] for r in con.execute("SELECT DISTINCT etiqueta_real FROM clasificaciones")}
    con.close()
    assert "No aprovechable (Multicapa)" in etiquetas
    assert "No aprovechable (Icopor)" in etiquetas
    assert None in etiquetas                         # Food waste: no evaluable


def test_idempotencia(coco_sintetico):
    _ingestar(coco_sintetico)
    antes = _contar()
    _ingestar(coco_sintetico)
    assert _contar() == antes                        # Reejecutar no duplica filas


def test_dry_run_no_escribe(coco_sintetico):
    _ingestar(coco_sintetico, "--dry-run")
    assert _contar() == 0


def test_borrar_origen(coco_sintetico):
    _ingestar(coco_sintetico)
    ip.main(["--borrar-origen", "taco_aug"])
    assert _contar("taco_aug") == 0 and _contar("taco") == 7


def test_solo_criticas_y_reporte_csv(coco_sintetico, tmp_path):
    csv = tmp_path / "r.csv"
    _ingestar(coco_sintetico, "--solo-criticas", "--dry-run", "--reporte-csv", str(csv))
    texto = csv.read_text(encoding="utf-8-sig")
    assert "Crisp packet" in texto and "Clear plastic bottle" not in texto


def test_exportar_recortes(coco_sintetico, tmp_path):
    destino = tmp_path / "recortes"
    _ingestar(coco_sintetico, "--dry-run", "--exportar-recortes", str(destino))
    carpetas = {p.name for p in destino.iterdir()}
    assert {"trash", "metal", "plastic"} <= carpetas
    assert not (destino / "None").exists()            # Food waste no se exporta


def test_salida_con_consola_cp1252(coco_sintetico, monkeypatch):
    """En Windows la consola redirigida usa cp1252: no debe fallar con emojis ni tildes."""
    import io
    import sys
    falso = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", falso)
    assert _ingestar(coco_sintetico, "--dry-run") == 0
    assert _contar() == 0
