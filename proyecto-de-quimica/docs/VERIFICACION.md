# Verificación del proyecto

**Fecha:** 7 de noviembre de 2026
**Resultado de las pruebas automáticas:** ✅ **53 de 53 pasan** (`python -m pytest -q`, 3,7 s)
**Entorno de verificación:** Python 3.13 · albumentations 2.0.8 · pycocotools 2.0.11 · OpenCV 4.x

## 1. Auditoría y correcciones realizadas

Una revisión anterior (hecha con un asistente local) dejó varios problemas que se corrigieron:

| # | Problema encontrado | Corrección |
|---|---|---|
| 1 | `ingesta_publica.py` había vuelto a una versión antigua (sin pycocotools ni albumentations) y se le quitaron las tildes a las etiquetas | Se restauró la versión correcta; prueba `test_destinos_validos_y_grupos_con_tilde` |
| 2 | Por el punto 1, la línea base reportaba 12,4 % y 0 % en plástico y papel | Recalculada: **32,7 %** (ver [RESULTADOS.md](../RESULTADOS.md)) |
| 3 | `reentrenar.py` tenía un «parche de compatibilidad» de 238 líneas con un mapeo de clases duplicado e inconsistente | Se restauró la versión original, que usa el mapeo único de `ingesta_publica.py` |
| 4 | Las imágenes `docs/img/diagnostico_recortes.png` y `ejemplo_aumentos.png` no mostraban recortes reales (rectángulos de colores y una foto completa de 18 MB) | Eliminadas; el script real es `scripts/diagnostico_recortes.py` |
| 5 | Pruebas mal planteadas: sumaban los logits como si fueran probabilidades (−0,006), compartían la base de datos entre pruebas y probaban un `HorizontalFlip` ajeno al proyecto | Pruebas reescritas (sección 2) |
| 6 | Errores de codificación en la consola de Windows (cp1252) con tildes y emojis | Los scripts fuerzan UTF-8 al iniciar; se eliminaron los envoltorios `forzar_utf8.py` y `reentrenar_wrapper.py` |
| 7 | Archivos temporales en el repositorio (`temp_recortes/` 13 MB, `.lmstudio/`, `descarga_temp/`, zips) | Eliminados y añadidos al `.gitignore` |

## 2. Pruebas automáticas (`tests/`)

Cada prueba usa **su propia base SQLite temporal** (`tests/conftest.py`): nunca se toca
`data/reciclaje.db` y las pruebas no se afectan entre sí. Los datos se generan de forma
sintética (no hace falta descargar TACO ni TrashNet).

| Archivo | Pruebas | Qué verifica |
|---|---:|---|
| `test_mapeo.py` | 14 | Clases críticas (papas → Multicapa, icopor → Icopor, aerosol → Metal), respaldo por súper-categoría, las 60 categorías de TACO tienen clase, grupos con tilde idénticos a los del motor |
| `test_ingesta.py` | 9 | Lectura COCO, orientación EXIF, inserción en modo WAL, idempotencia (reejecutar no duplica), `--dry-run`, `--borrar-origen`, `--solo-criticas`, `--exportar-recortes`, consola cp1252 |
| `test_aumentos.py` | 6 | Reproducibilidad con semilla, variación con semillas distintas, presencia de RandomCrop / RandomSunFlare / RandomBrightnessContrast, más destellos en empaques metalizados, tamaños |
| `test_database.py` | 14 | `grupo_material`, duplicados, validaciones, funciones del tablero, lógica de validación por grupo |
| `test_modelo.py` | 6 | Softmax correcto (≥ 0 y suma 1 con tolerancia 1e-5), carga del modelo, los 4 motores |
| `test_reentrenar.py` | 4 | Red base + capa final = modelo original (diferencia < 0,001), exportación de un modelo nuevo, regla de decisión |

```bash
python -m pytest -q          # resumen
python -m pytest -q -rA      # con el nombre de cada prueba
```

## 3. Verificaciones que debes hacer en tu computador (necesitan TACO)

Estas no pueden correr en las pruebas automáticas porque requieren el dataset real:

1. **Diagnóstico visual de recortes** — confirma que cada recorte muestra el objeto correcto:
   ```bash
   python scripts/diagnostico_recortes.py --ruta <TACO/data> --aumentados
   ```
   Abre `docs/img/diagnostico_recortes.jpg`. Cada imagen debe mostrar el objeto rotulado.
2. **Línea base con la versión actual** (para comparar con el reentrenamiento):
   ```bash
   python ingesta_publica.py --ruta <TACO/data> --limite 800 --dry-run --reporte-csv docs/resultados/linea_base_taco_v2.csv
   ```
3. **Reentrenamiento** (10-25 min la primera vez):
   ```bash
   python entrenamiento/reentrenar.py --taco <TACO/data>
   ```
   Solo reemplaza el modelo si mejora; el reporte queda en `models/reporte_reentrenamiento.txt`.

## 4. Archivos que NO se suben a GitHub

`TACO/`, `entrenamiento/dataset-resized/`, `entrenamiento/cache/`, `entrenamiento/salidas/`,
`models/respaldo/`, `data/` (base de datos), `.venv/`, `temp_recortes/`, `para_revision/`, `*.zip`.
