# Guía de reentrenamiento de la IA con fotos propias

**Proyecto:** Reciclaje Inteligente · Pipeline de 2 etapas (detector + clasificador)
**Para quién:** cualquier persona del equipo que quiera agregar fotos nuevas y volver a entrenar el modelo.
**Sistema:** Windows (PowerShell). En macOS/Linux los comandos de Python son iguales.

> Todos los comandos se ejecutan **dentro de la carpeta `proyecto-de-quimica`** (la que contiene `app.py`)
> y con el entorno virtual activado (`.\venv\Scripts\Activate.ps1`).

---

## Índice

1. [Cómo aprende la IA de tus fotos](#1-cómo-aprende-la-ia-de-tus-fotos)
2. [Estructura de carpetas](#2-estructura-de-carpetas)
3. [Nombres de carpeta válidos](#3-nombres-de-carpeta-válidos)
4. [Cómo tomar buenas fotos](#4-cómo-tomar-buenas-fotos)
5. [Primera vez: preparación (solo una vez)](#5-primera-vez-preparación-solo-una-vez)
6. [Agregar fotos nuevas y reentrenar (rutina)](#6-agregar-fotos-nuevas-y-reentrenar-rutina)
7. [Qué hace el entrenamiento, etapa por etapa](#7-qué-hace-el-entrenamiento-etapa-por-etapa)
8. [Cómo leer los reportes](#8-cómo-leer-los-reportes)
9. [Respaldar y restaurar modelos](#9-respaldar-y-restaurar-modelos)
10. [Subir los cambios al repositorio](#10-subir-los-cambios-al-repositorio)
11. [Problemas frecuentes](#11-problemas-frecuentes)
12. [Lista de verificación rápida](#12-lista-de-verificación-rápida)

---

## 1. Cómo aprende la IA de tus fotos

La IA **no** etiqueta tus fotos sola. Tú le dices qué es cada foto **poniéndola en la carpeta correcta**:
el nombre de la carpeta es la etiqueta. Con muchas fotos etiquetadas, la IA aprende a reconocer cada
material; ya entrenada, es ella la que clasifica en vivo lo que ve la cámara.

Hay dos niveles de carpetas, como cajones dentro de cajones:

| Nivel | Qué representa | Ejemplo |
|---|---|---|
| 1. **Sesión** | Dónde y cuándo tomaste las fotos | `cocina_tarde`, `sala_noche` |
| 2. **Clase** | Qué objeto aparece en la foto | `bolsa_blanca`, `servilleta`, `icopor` |

**Regla de oro:** nunca mezcles objetos distintos en la misma carpeta de clase.

---

## 2. Estructura de carpetas

Crea **una sola vez** la carpeta `fotos_propias` dentro de `proyecto-de-quimica`.
Esta carpeta **no se sube a GitHub** (está en el `.gitignore`).

```
proyecto-de-quimica/
├── app.py
├── fotos_propias/                 ← crear una sola vez
│   ├── cocina_tarde/              ← SESIÓN (lugar + momento). Nombre libre, sin espacios
│   │   ├── bolsa_blanca/          ← CLASE. Nombre de la tabla de la sección 3
│   │   │   ├── IMG_2031.jpg       ← nombre de archivo libre, no hay que enumerar
│   │   │   ├── IMG_2032.jpg
│   │   │   └── ...
│   │   ├── servilleta/
│   │   ├── icopor/
│   │   └── fondo/
│   ├── sala_noche/                ← otra sesión, otro día
│   │   ├── bolsa_blanca/
│   │   └── paquete_papas/
│   └── prueba_oficina/            ← SESIÓN DE EXAMEN (empieza por "prueba")
│       ├── bolsa_blanca/
│       ├── servilleta/
│       └── ...
└── ...
```

### Reglas

| Elemento | Regla |
|---|---|
| Nombre de sesión | Libre, sin espacios: `cocina_tarde`, `patio_dia`, `colegio_lunes` |
| Sesión de examen | Su nombre **debe empezar por `prueba`**: `prueba_oficina`, `prueba_casa2` |
| Nombre de clase | **Exactamente** uno de la sección 3 (mayúsculas y tildes no importan) |
| Archivos | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. Nombre libre: déjalos como salen del celular |
| Sesiones distintas | No tienen que tener las mismas clases |

### ¿Para qué sirve la sesión `prueba_...`?

La IA **no aprende** de esas fotos. Se usan al final como **examen**: fotos que nunca vio, tomadas
con tu cámara y tu luz. Así sabes si el modelo funciona de verdad en tu entorno.

- Haz al menos **una** sesión de prueba con los mismos tipos de objeto que fotografiaste para entrenar.
- **No renombres** una sesión `prueba_` después de entrenar: el examen dejaría de ser comparable.
- No muevas fotos de una sesión de prueba a una de entrenamiento (sería "darle las respuestas del examen").

### Crear las carpetas automáticamente (PowerShell)

Cambia el nombre de la sesión y la lista de clases según lo que vayas a fotografiar:

```powershell
$sesion = "cocina_tarde"
"bolsa_blanca","servilleta","icopor","paquete_papas","pet","fondo" |
    ForEach-Object { New-Item -ItemType Directory -Force "fotos_propias\$sesion\$_" | Out-Null }
Get-ChildItem "fotos_propias\$sesion"
```

Luego solo arrastra cada foto a su carpeta.

---

## 3. Nombres de carpeta válidos

La lista completa está en `taxonomia/master.yaml` (líneas `propias:` de cada clase).

| Carpeta (cualquiera de estos nombres) | Clase de la IA | Caneca |
|---|---|---|
| `pet`, `botella_pet`, `botella_plastica` | Botella PET | ⚪ Blanca |
| `pead`, `pp`, `envase_aseo`, `galon`, `tarrina`, `detergente`, `cloro` | Envase PEAD / PP | ⚪ Blanca (sucio → ⚫) |
| `tapa_plastica`, `vaso_plastico`, `cubiertos`, `pitillo` | Otro plástico rígido | ⚪ Blanca (sucio → ⚫) |
| `bolsa`, `bolsa_blanca`, `bolsa_plastica`, `pelicula`, `plastico_flexible` | Bolsa / película plástica | ⚪ Blanca (sucia → ⚫) |
| `paquete_papas`, `empaque_snack`, `empaque_galletas`, `multicapa`, `metalizado`, `crema_dental` | Multicapa flexible | ⚫ Negra |
| `tetra_pak`, `tetrapak`, `caja_leche` | Tetra Pak | ⚪ Blanca |
| `icopor`, `poliestireno`, `bandeja_icopor`, `coca_icopor`, `vaso_icopor` | Icopor | ⚫ Negra |
| `vidrio`, `botella_vidrio`, `frasco` | Vidrio | ⚪ Blanca |
| `lata`, `lata_aluminio`, `lata_atun` | Lata | ⚪ Blanca |
| `aerosol`, `desodorante_aerosol`, `papel_aluminio`, `tapa_metalica` | Otro metal | ⚪ Blanca (sucio → ⚫) |
| `papel`, `hoja`, `periodico`, `revista` | Papel | ⚪ Blanca (sucio → ⚫) |
| `carton`, `caja`, `caja_carton` | Cartón | ⚪ Blanca (sucio → ⚫) |
| `servilleta`, `papel_higienico`, `tapabocas`, `colilla`, `guante` | Sanitario / rechazo | ⚫ Negra |
| `organico`, `cascara`, `restos_comida` | Orgánico | 🟢 Verde |
| `pila`, `bateria`, `bombillo`, `medicamento` | Peligroso | Punto posconsumo |
| `fondo`, `vacio`, `sin_objeto` | Sin objeto | — |

### ¿Y si quiero usar un nombre que no está?

Las fotos no se pierden ni rompen nada: aparecen como **"sin mapeo"** en el reporte del manifiesto y
no se usan. Para que cuenten, agrega el nombre en `taxonomia/master.yaml`, en la línea `propias:`
de la clase correcta. Ejemplo, para aceptar la carpeta `bolsa_mercado`:

```yaml
  - id: 3
    nombre: plastico_flexible_bolsa
    ...
    alias:
      ...
      propias: [bolsa, bolsa_plastica, bolsa_blanca, pelicula, plastico_flexible, bolsa_mercado]
```

Después ejecuta `python -m taxonomia` (debe mostrar las 16 clases sin errores) y `python -m pytest -q`.

> ⚠️ No cambies los `id` ni el orden de las clases en `master.yaml`: el modelo entrenado depende de ellos.

---

## 4. Cómo tomar buenas fotos

| Hacer ✅ | Evitar ❌ |
|---|---|
| Un solo objeto por foto | Varios residuos en la misma foto |
| El objeto ocupa buena parte del encuadre | Objeto diminuto en una esquina |
| Objetos **distintos** (varias bolsas, marcas y tamaños) | 50 fotos de la misma bolsa |
| Cambiar fondo, luz y ángulo entre fotos | Siempre la misma mesa y la misma luz |
| Usar la cámara con la que funcionará el sistema | Solo fotos de internet |
| Objetos como llegan a la caneca (arrugados, aplastados) | Solo productos nuevos y perfectos |
| Fotos de `fondo` (mesa, piso, caneca **sin** objeto) | Olvidar la clase `fondo` |

**Importante:** tus fotos se usan **completas, sin recorte**. Deben verse como ve la cámara del sistema.

### ¿Cuántas fotos?

| Meta | Por clase y por sesión | Total por clase |
|---|---|---|
| Mínimo para empezar | 15 – 25 | ~60 (3 sesiones + 1 de prueba) |
| Recomendado | 25 – 40 | 100 – 150 |

**Prioridad** (son las que hoy se confunden): `bolsa_blanca`, `servilleta`, `icopor`, `paquete_papas`,
`pet`, `fondo`. Luego `tetra_pak`, `pead`, `lata`, `carton`.

---

## 5. Primera vez: preparación (solo una vez)

Si ya lo hiciste, salta a la sección 6.

```powershell
# 1. Dependencias de la app y de las pruebas
pip install -r requirements.txt
python -m pytest -q                       # deben pasar todas las pruebas

# 2. Dependencias de entrenamiento (PyTorch, Ultralytics, ONNX...)
pip install -r entrenamiento/requirements-entrenamiento.txt

# 3. Pesos preentrenados (ÚNICO paso que necesita internet)
python entrenamiento/train_pipeline.py --etapas preparar-pesos
```

El paso 3 deja `yolo11n.pt` y los pesos de MobileNetV3 en `entrenamiento/pesos_base/`. Después,
todo el entrenamiento funciona sin conexión.

**TACO (muy recomendado):** descárgalo en una carpeta **fuera** del proyecto
(`https://github.com/pedropro/TACO`, script `download.py`) para que quede en `..\TACO\data`.
Sin TACO el detector no tiene con qué entrenarse y el clasificador tiene muchos menos ejemplos.

---

## 6. Agregar fotos nuevas y reentrenar (rutina)

Esta es la rutina cada vez que tomes fotos nuevas (mañana, la semana que viene...).

> ⚠️ Si hay un entrenamiento corriendo, **puedes copiar fotos** a `fotos_propias` sin problema,
> pero **no ejecutes los pasos B a E** hasta que termine.

### Paso A — Guardar las fotos

Opción recomendada: **una sesión nueva** por cada día o lugar.

```
fotos_propias\patio_dia\bolsa_blanca\IMG_3001.jpg
fotos_propias\patio_dia\paquete_papas\IMG_3010.jpg
```

También puedes agregarlas a una sesión existente (`fotos_propias\cocina_tarde\servilleta\...`):
funciona igual, y esas fotos quedan en el mismo grupo que las anteriores de esa sesión.

### Paso B — Reconstruir el manifiesto

El manifiesto es la lista de todas las fotos con su clase y su conjunto (entrenamiento, validación
o prueba). **Se rehace desde cero cada vez**: vuelve a leer todas las carpetas, viejas y nuevas,
así que no hay que indicarle cuáles fotos son nuevas.

```powershell
# Con TACO (recomendado)
python datos/construir_manifiesto.py --taco ..\TACO\data --propias fotos_propias --salida data/manifiesto.parquet

# Solo con tus fotos
python datos/construir_manifiesto.py --propias fotos_propias --salida data/manifiesto.parquet
```

Si en el entrenamiento original usaste otras fuentes (`--trashnet`, `--zerowaste`,
`--off-productos`), **inclúyelas siempre** en este comando: el manifiesto solo contiene lo que le pases.

### Paso C — Revisar el reporte del manifiesto

Abre `data\manifiesto_reporte.txt` y verifica:

1. Que tus clases aparezcan en la tabla **"objetos por clase y conjunto"** y que los números hayan subido.
2. Que tus fotos de las sesiones `prueba_...` estén en la columna **prueba**.
3. Que **no** aparezca ninguna carpeta tuya en **"Etiquetas SIN mapeo"** (si aparece, revisa la sección 3).

### Paso D — Respaldar los modelos actuales

Cada entrenamiento **sobrescribe** los archivos de `models/`, aunque el modelo nuevo salga peor.
Haz siempre una copia antes (ver sección 9):

```powershell
$fecha = Get-Date -Format "yyyyMMdd_HHmm"
New-Item -ItemType Directory -Force "models\respaldo\$fecha" | Out-Null
Copy-Item models\*.onnx, models\*.json, models\*.txt "models\respaldo\$fecha"
```

### Paso E — Reentrenar

**Caso normal: solo agregaste fotos propias** → reentrena **solo el clasificador**.
Tus fotos no tienen cajas dibujadas, así que no le aportan nada al detector; reentrenarlo sería
perder horas.

```powershell
python entrenamiento/train_pipeline.py --solo-clasificador
```

**Caso especial: agregaste un dataset con cajas** (otra versión de TACO, ZeroWaste...) → todo:

```powershell
python entrenamiento/train_pipeline.py
```

Comandos útiles:

| Comando | Para qué |
|---|---|
| `python entrenamiento/train_pipeline.py --dry-run` | Ver el plan y los conteos **sin entrenar** |
| `python entrenamiento/train_pipeline.py --rapido` | Prueba de humo: 1 época, pocas imágenes (minutos) |
| `python entrenamiento/train_pipeline.py --solo-clasificador` | Reentrenar solo el clasificador |
| `python entrenamiento/train_pipeline.py --solo-detector` | Reentrenar solo el detector |
| `python entrenamiento/train_pipeline.py --etapas calibrar,exportar-clasificador,evaluar` | Repetir etapas concretas |

### Paso F — Comparar y decidir

Abre `models\reporte_pipeline.txt` y compáralo con el del respaldo
(`models\respaldo\<fecha>\reporte_pipeline.txt`), sobre todo la **exactitud balanceada**, el
**recall** de las clases críticas y la **contaminación de la Blanca** (sección 8).

- Si mejoró → súbelo al repositorio (sección 10).
- Si empeoró → restaura el respaldo (sección 9).

---

## 7. Qué hace el entrenamiento, etapa por etapa

Toda la configuración está en `entrenamiento/pipeline.yaml`.

### Etapas del DETECTOR (encuentra *dónde* hay un residuo)

**1. `exportar-detector`**
- **Carga:** del manifiesto, las fotos de TACO/ZeroWaste que tienen cajas dibujadas.
- **Hace:** borra y rearma `entrenamiento/trabajo/detector/` en el formato de YOLO (imágenes,
  archivos `.txt` con las cajas y `data.yaml`). Todas las cajas se llaman igual: `residuo`.

**2. `entrenar-detector`**
- **Carga:** `entrenamiento/pesos_base/yolo11n.pt`, un modelo que ya sabe detectar objetos en general.
- **Hace:** lo especializa en basura, hasta 100 vueltas (épocas) sobre las fotos.
- **Verifica:** después de cada época se mide con las fotos de validación y guarda la mejor versión.
  Si pasan 20 épocas sin mejorar, se detiene solo.
- **Produce:** `models/detector_residuo.onnx` + `models/detector_residuo.json`.

### Etapas del CLASIFICADOR (dice *qué material* es)

**3. `exportar-recortes`**
- **Carga:** del manifiesto, todos los objetos con clase conocida.
- **Hace:** recorta cada caja (con un margen del 15 %); tus fotos se usan completas. Las guarda a
  256 px en `entrenamiento/trabajo/recortes/<conjunto>/<id_clase>/`. Además genera recortes de
  **fondo** desde zonas de TACO donde no hay basura.
- **Verifica:** descarta cajas menores a 32 px (no se distingue el material) e informa imágenes dañadas.
- **Ahorra tiempo:** los recortes que ya existen no se rehacen.

**4. `entrenar-clasificador`** (la etapa más larga)
- **Balancea:** máximo 3 000 fotos por clase en entrenamiento; Open Food Facts y TrashNet no pueden
  pasar del 30 % de una clase.
- **Carga:** MobileNetV3-Large con pesos de ImageNet. **No parte del modelo anterior:** cada
  reentrenamiento empieza de cero y aprende de todas las fotos, viejas y nuevas.
- **Entrena todas las capas.** En cada época varía las fotos (gira, cambia la luz, las deforma como
  arrugadas, desenfoca, tapa pedazos, mezcla dos fotos) para que aprenda el material y no la foto exacta.
- **Verifica:** después de cada época mide la **exactitud balanceada** en validación (el promedio
  de aciertos de cada clase, para que una clase fácil no tape los fallos de otra). Guarda la mejor
  versión; se detiene si pasan 6 épocas sin mejorar o al llegar a 30.
- **Avisa** si alguna de las 16 clases no tiene fotos.

**5. `calibrar`**
- Ajusta un número llamado **temperatura** sobre las fotos de validación para que la confianza
  signifique algo: cuando dice 80 %, que acierte cerca del 80 % de las veces.

**6. `exportar-clasificador`**
- **Produce:** `models/clasificador_material.onnx` (con la calibración incluida) y
  `models/clasificador_material.json` (nombres de las 16 clases, temperatura, tamaño de entrada).
- **Verifica:** que el ONNX dé los mismos resultados que el modelo original en onnxruntime y en OpenCV.

**7. `evaluar`** (el examen final)
- **Carga:** fotos que el modelo nunca vio: tus sesiones `prueba_...` y los lotes 14 y 15 de TACO.
- **Mide:** aciertos por clase y en total, exactitud balanceada, caneca correcta (usando
  `taxonomia/canecas.yaml`) y **contaminación de la Blanca** (cuánto de lo que mandó a la Blanca no
  debía ir ahí).
- **Criterios de aprobación** (`pipeline.yaml` → `evaluacion`):
  - bolsa, servilleta, icopor, paquete de papas y PET aciertan **≥ 80 %** cada una;
  - contaminación de la Blanca **≤ 10 %**;
  - exactitud balanceada **≥ 70 %**.
- **Produce:** `models/reporte_pipeline.txt` y `entrenamiento/trabajo/clasificador/evaluacion.json`.

> ⚠️ El resultado **ACEPTADO / NO ACEPTADO es informativo**: no bloquea ni deshace la exportación.
> Por eso existe el paso D (respaldo).

### ¿Y `entrenamiento/reentrenar.py`?

Es el **script viejo**: solo ajusta la última capa del modelo de 6 clases
(`models/trashnet_mobilenetv2.onnx`), que es el que usa la app hoy. El pipeline nuevo es
independiente y no lo toca. **No los mezcles.** La app seguirá usando el modelo viejo hasta que se
conecten los dos modelos nuevos (tarea pendiente).

---

## 8. Cómo leer los reportes

### `data/manifiesto_reporte.txt` (paso C)

```
Clasificador · objetos por clase y conjunto:
split                                   entrenamiento  validacion  prueba  All
clase_id clase_master
3        plastico_flexible_bolsa                  412          61      48  521
12       sanitario_y_rechazo                      380          55      40  475
...
Etiquetas SIN mapeo (agrégalas a taxonomia/master.yaml):
  propias    bolsa_mercado                                                    35
```

- La columna **prueba** debe tener fotos de cada clase importante (si está en 0, falta una sesión `prueba_`).
- Cualquier línea `propias` en **SIN mapeo** es una carpeta mal nombrada (sección 3).

### `models/reporte_pipeline.txt` (paso F)

```
Evaluación en PRUEBA · 612 recortes
  Exactitud:              78.4%
  Exactitud balanceada:   74.1%
  Caneca correcta:        86.9%
  Contaminación Blanca:    8.2%
  Recall por clase:
    icopor                      83.3%
    plastico_flexible_bolsa     81.0%
    sanitario_y_rechazo         72.5%
    ...
❌ sanitario_y_rechazo: recall 72.5% < 80%
=> MODELO NO ACEPTADO
```

*(Números de ejemplo.)* Cómo interpretarlo:

| Indicador | Qué significa | Si está mal |
|---|---|---|
| Recall de una clase | De cada 100 objetos de esa clase, cuántos reconoció | Toma más fotos de **esa** clase, variadas |
| Contaminación Blanca | Qué parte de lo enviado a la Blanca no era aprovechable | Más fotos de servilleta, icopor y paquetes |
| Exactitud balanceada | Promedio de aciertos de todas las clases | Revisa qué clases bajan el promedio |
| Por fuente | Aciertos separados en TACO y en tus fotos | Si `propias` está bajo, faltan fotos en tu entorno |

---

## 9. Respaldar y restaurar modelos

`models/respaldo/` está en el `.gitignore`: los respaldos se quedan en tu computador.

**Respaldar (antes de cada entrenamiento):**

```powershell
$fecha = Get-Date -Format "yyyyMMdd_HHmm"
New-Item -ItemType Directory -Force "models\respaldo\$fecha" | Out-Null
Copy-Item models\*.onnx, models\*.json, models\*.txt "models\respaldo\$fecha"
Get-ChildItem "models\respaldo"          # lista los respaldos
```

**Restaurar (si el modelo nuevo salió peor):**

```powershell
Get-ChildItem "models\respaldo"          # elige la carpeta, por ejemplo 20261108_1930
Copy-Item "models\respaldo\20261108_1930\*" models\ -Force
```

**Liberar espacio:** `entrenamiento\trabajo\` se puede borrar cuando quieras (se regenera solo,
aunque la siguiente vez los recortes tardan más). `entrenamiento\pesos_base\` **no** lo borres:
volver a crearlo requiere internet.

---

## 10. Subir los cambios al repositorio

### Qué se sube y qué no

| Se sube ✅ | No se sube ❌ (ya está en `.gitignore`) |
|---|---|
| `models/detector_residuo.onnx` y `.json` | `fotos_propias/` (tus fotos) |
| `models/clasificador_material.onnx` y `.json` | `entrenamiento/trabajo/` (recortes, corridas) |
| `models/reporte_pipeline.txt` | `entrenamiento/pesos_base/` (pesos descargados) |
| `taxonomia/*.yaml` (si los cambiaste) | `models/respaldo/` |
| Esta guía y cualquier código modificado | `data/` (manifiesto y base de datos) |

Los modelos pesan unos 10–25 MB cada uno; GitHub admite archivos de hasta 100 MB.

### Pasos

```powershell
cd ..                                      # raíz del repositorio (donde está la carpeta .git)
git status                                 # revisa qué cambió: NO deben aparecer fotos ni "trabajo/"
git add proyecto-de-quimica/models/detector_residuo.onnx proyecto-de-quimica/models/detector_residuo.json
git add proyecto-de-quimica/models/clasificador_material.onnx proyecto-de-quimica/models/clasificador_material.json
git add proyecto-de-quimica/models/reporte_pipeline.txt
git add proyecto-de-quimica/docs/GUIA_REENTRENAMIENTO.md
git commit -m "Reentrenamiento 2026-11-08: +120 fotos propias (bolsa, servilleta, icopor)"
git push
```

**Buena práctica:** en el mensaje del commit escribe qué fotos agregaste y la exactitud balanceada
del reporte. Así el historial de Git sirve como bitácora de cada versión del modelo.

**Si `git status` muestra fotos o carpetas de `trabajo/`:** no las agregues. Revisa que el
`.gitignore` tenga las líneas `fotos_propias/` y `entrenamiento/trabajo/`.

---

## 11. Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `❌ No existe data/manifiesto.parquet` | No ejecutaste el paso B | Ejecuta `datos/construir_manifiesto.py` |
| Mis fotos aparecen en "SIN mapeo" | Nombre de carpeta no reconocido | Renombra la carpeta (sección 3) o agrega el alias |
| `⚠️ Clases SIN ejemplos de entrenamiento` | Ninguna foto de esa clase en entrenamiento | Toma fotos de esa clase en una sesión que **no** sea `prueba_` |
| `⚠️ No hay recortes de prueba` | No hay sesión `prueba_...` | Crea una sesión `prueba_...` y repite desde el paso B |
| La columna **prueba** está en 0 para una clase | La sesión de prueba no tiene esa clase | Agrega esa clase dentro de tu sesión `prueba_...` |
| `Falta el dataset del detector` | Usaste `--etapas` sin `exportar-detector` | Ejecuta sin `--etapas` o incluye `exportar-detector` |
| Ultralytics intenta descargar algo y falla | Falta `yolo11n.pt` en `pesos_base` | Con internet: `--etapas preparar-pesos` |
| Se queda sin memoria (RAM/VRAM) | Lote demasiado grande | En `pipeline.yaml` baja `batch` (64 → 32 → 16) y `workers` (4 → 2) |
| Muy lento sin GPU | Normal en CPU | Usa `--solo-clasificador`; déjalo correr de noche |
| Cambié `margen` o `lado` en `pipeline.yaml` | Los recortes viejos no se rehacen | Borra `entrenamiento\trabajo\recortes` antes de entrenar |
| El modelo nuevo es peor | Fotos poco variadas o mal clasificadas | Restaura el respaldo (sección 9) y revisa las carpetas |

---

## 12. Lista de verificación rápida

```
[ ] Fotos en fotos_propias\<sesion>\<clase>\ con nombres de la sección 3
[ ] Al menos una sesión prueba_... con las clases importantes
[ ] Ningún entrenamiento corriendo
[ ] Paso B: manifiesto reconstruido (con TODAS las fuentes de siempre)
[ ] Paso C: reporte sin "SIN mapeo" y con fotos en la columna prueba
[ ] Paso D: respaldo de models\ hecho
[ ] Paso E: python entrenamiento/train_pipeline.py --solo-clasificador
[ ] Paso F: reporte_pipeline.txt comparado con el respaldo
[ ] Si mejoró: git add (solo modelos/reportes/docs) + commit + push
[ ] Si empeoró: restaurar respaldo
```
