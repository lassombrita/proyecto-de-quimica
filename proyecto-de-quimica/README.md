# ♻️ Reciclaje Inteligente

**Plataforma de clasificación de residuos con Visión por Computador y tablero analítico en tiempo real.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67%2B-FF4B4B?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactivo-3F4F75?logo=plotly&logoColor=white)

---

## Descripción

**Reciclaje Inteligente** es una solución para organizaciones (empresas, campus, centros comerciales, conjuntos residenciales y operadores de aseo) que necesitan mejorar la separación en la fuente y medir el desempeño de sus programas de reciclaje.

A partir de una imagen capturada por cámara, el sistema identifica el tipo de residuo, indica el contenedor de destino según la normativa colombiana y registra cada evento para su análisis.

### Funcionalidades principales

| Módulo | Descripción |
|---|---|
| 📷 **Clasificador en vivo** | Captura desde la cámara web con control explícito de encendido/apagado, clasificación en seis categorías y resultado visual inmediato. |
| 🗂️ **Asignación normativa** | Contenedor de destino según la **Resolución 2184 de 2019** (código de colores blanco, negro y verde). |
| 📊 **Tablero analítico** | Indicadores clave, distribución por material, volumen por contenedor e historial interactivo. |
| 🗄️ **Trazabilidad** | Cada clasificación queda registrada con fecha, hora, material, nivel de confianza y contenedor. |
| 🔒 **Privacidad** | La cámara permanece apagada por defecto y se apaga automáticamente al salir del módulo de captura. |

### Categorías soportadas

| Categoría | Ejemplos | Contenedor |
|---|---|---|
| PET (resina #1) | Botellas de bebidas (solo con cuello visible) | ⚪ Blanco |
| PEAD (resina #2) | Envases de detergente y lácteos (pico lateral visible) | ⚪ Blanco |
| Plástico | Plástico sin evidencia de resina: tapas, vasos, envases | ⚪ Blanco |
| Vidrio | Botellas y frascos | ⚪ Blanco |
| Metal | Latas de aluminio y acero | ⚪ Blanco |
| Papel/Cartón | Cajas, hojas, periódico | ⚪ Blanco |
| No aprovechable | El modelo actual agrupa aquí película plástica, icopor, multicapa y sanitarios | ⚫ Negro |

> Estas son las clases del modelo actual (v1). La migración a la **Taxonomía Maestra de 16 clases**
> (`taxonomia/master.yaml`) y a las reglas de canecas configurables (`taxonomia/canecas.yaml`)
> está descrita en la sección *Pipeline de 2 etapas*.

> El contenedor ⚪ blanco recibe los residuos aprovechables, el ⚫ negro los no aprovechables y el 🟢 verde los orgánicos aprovechables.

---

## Arquitectura

La aplicación sigue una **arquitectura en capas**: cada módulo tiene una sola responsabilidad y puede reemplazarse sin afectar a los demás.

```
┌──────────────────────────────────────────────────────────┐
│  CAPA DE PRESENTACIÓN — app.py + assets/styles.css       │
│  Streamlit · Plotly · navegación · selector multi-engine │
└───────────────┬───────────────────────────┬──────────────┘
                │                           │
┌───────────────▼──────────────┐ ┌──────────▼───────────────┐
│  CAPA DE INFERENCIA          │ │  CAPA DE DATOS           │
│  vision_engine.py            │ │  database.py             │
│  (wrapper: vision_mock.py)   │ │  SQLite · Pandas         │
│  OpenCV · Pillow · NumPy     │ │  eventos_reciclaje       │
│  ClasificadorResiduos        │ │                          │
└──────────────────────────────┘ └──────────────────────────┘
```

### Estructura del repositorio

```
.
├── app.py                 # Interfaz, navegación, selector de motor y diagnóstico
├── vision_engine.py       # Motor central multi-engine (Color, Geometría, IA, Híbrido)
├── vision_mock.py         # Enrutador/wrapper de compatibilidad hacia vision_engine.py
├── ingesta_publica.py     # Ingesta COCO (TACO/Roboflow) + aumento de datos -> SQLite
├── RESULTADOS.md          # Línea base del modelo frente a TACO
├── scripts/
│   └── diagnostico_recortes.py     # Cuadrícula visual de recortes TACO
├── tests/                 # Pruebas automáticas (pytest)
├── docs/
│   ├── AUDITORIA_Y_CORRECCION_VISION.md  # Informe técnico de auditoría y correcciones
│   ├── VERIFICACION.md             # Auditoría y pruebas
│   ├── GUIA_EXPOSICION.md          # Guion y preguntas para la exposición
│   └── resultados/                 # CSV de las mediciones
├── models/
│   ├── trashnet_mobilenetv2.onnx   # Red neuronal entrenada (se ejecuta con OpenCV)
│   ├── reporte_entrenamiento.txt   # Métricas de exactitud
│   └── licencias/                  # Licencias de los datos y pesos usados
├── entrenamiento/         # Convertir, entrenar, evaluar y reentrenar (reentrenar.py)
├── database.py            # Persistencia SQLite
├── requirements.txt       # Dependencias
├── assets/
│   └── styles.css         # Sistema visual (tarjetas, métricas, tema)
├── .streamlit/
│   └── config.toml        # Tema y configuración del servidor
└── data/
    └── reciclaje.db       # Base de datos (se genera automáticamente)
```

### Motor de clasificación (Multi-Engine)

El sistema integra **2 motores de producción** (IA e Híbrido) y **2 motores didácticos** (Color y
Geometría) que reproducen fallos conocidos para la exposición. Los didácticos solo aparecen con:

```bash
streamlit run app.py -- --modo-didactico      # o RECICLAJE_MODO_DIDACTICO=1
```

Sus eventos se guardan con origen `camara_didactico` y no se mezclan con las estadísticas reales.

1. **Motor Básico (Color)** *(didáctico)*: Heurística cromática en HSV. Demuestra el sesgo cromático (ej. botella PET con jugo café clasificada como cartón). Ya **no** convierte lo blanco en PET: un tono claro sin saturación da «No identificado».
2. **Motor Geométrico (Forma/Contornos)** *(didáctico)*: Clasificación basada puramente en OpenCV (Canny, descriptores de aspecto, solidez, rectangularidad y análisis de cuello). No depende de color.
3. **Motor IA (MobileNetV2):** Inferencia directa con la red neuronal convolucional entrenada en TrashNet y ejecutada con `cv2.dnn` sobre canales RGB. Sin bloqueos geométricos.
4. **Motor Híbrido Estricto:** La red neuronal decide el material; la forma solo especializa «plástico» a PET/PEAD cuando hay un cuello real (un extremo angosto, el otro ancho y figura alargada: un óvalo como una bolsa arrugada ya no cuenta). No suma confianza. Si la 1.ª y la 2.ª clase están a menos de 15 puntos, responde «No identificado».

> Para el análisis detallado de la auditoría y causas raíz de fallos previos, consulta el [Informe de Auditoría y Corrección de Visión](docs/AUDITORIA_Y_CORRECCION_VISION.md).

#### Política de incertidumbre (Fallback a contenedor Negro)
Si la confianza calculada es menor al **50.0%**, el sistema no adivina: devuelve **No identificado** y sugiere el contenedor **Negro** (Resolución 2184 de 2019), evitando así la contaminación de los materiales reciclables.

#### Modo Diagnóstico (Auditoría en Vivo)
La interfaz incluye un interruptor en la barra lateral para **Activar Modo Diagnóstico**. Al habilitarlo, muestra debajo del resultado:
- **Detector de formas:** Relación de aspecto, rectangularidad, solidez, ratio de cuello y tipo de silueta.
- **Red neuronal:** Top 3 de predicciones con porcentajes exactos y barras de progreso.
- **Análisis cromático:** Coordenadas HSV y color hexadecimal medido (cuando aplique).

### Modelo de red neuronal

| Elemento | Detalle |
|---|---|
| Arquitectura | MobileNetV2 (ImageNet) + capa de clasificación de 6 clases |
| Técnica | *Transfer learning* (extracción de características + regresión logística) |
| Datos | TrashNet: 2 527 fotos (cartón, vidrio, metal, papel, plástico, basura) |
| División | 70 % entrenamiento (con aumento de datos) · 15 % validación · 15 % prueba |
| Archivo | `models/trashnet_mobilenetv2.onnx` (8,9 MB) |

**Resultados en 380 fotos de prueba (nunca vistas durante el entrenamiento):**

| Métrica | Valor |
|---|---|
| Red neuronal sola: exactitud | 89,2 % |
| Sistema completo: objetos identificados | 68,7 % |
| Sistema completo: exactitud sobre los identificados | 92,3 % |
| Sistema completo: errores sobre el total | 5,3 % |

El sistema prefiere decir "No identificado" antes que arriesgar un error que contamine el contenedor blanco. El detalle por clase está en `models/reporte_entrenamiento.txt`.

**Reproducir el entrenamiento** (opcional; el modelo ya viene incluido):

```bash
pip install -r entrenamiento/requirements-entrenamiento.txt
python entrenamiento/convertir_mobilenet.py mobilenet_v2_deploy.prototxt mobilenet_v2.caffemodel backbone.onnx
python entrenamiento/entrenar_trashnet.py dataset-resized backbone.onnx models/trashnet_mobilenetv2.onnx
python entrenamiento/evaluar_sistema.py dataset-resized
```

Para esto se necesitan los pesos de [MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe) y `data/dataset-resized.zip` de [TrashNet](https://github.com/garythung/trashnet).

**Limitaciones:**
- TrashNet son fotos de objetos sobre fondo blanco. Con la cámara web, los mejores resultados se obtienen con el objeto completo dentro del encuadre, sobre un fondo liso y claro, y con buena iluminación.
- TrashNet no separa PET de PEAD; esa distinción la aporta la forma del pico.

**Créditos y licencias:** TrashNet © Gary Thung y Mindy Yang (MIT). Pesos de MobileNetV2 © Shicai Yang (BSD-3). Los textos de las licencias están en `models/licencias/`.

### Modelo de datos

Tabla `eventos_reciclaje`:

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | INTEGER (PK) | Identificador autoincremental |
| `timestamp` | TEXT (ISO 8601) | Fecha y hora del evento |
| `material` | TEXT | Categoría detectada |
| `confianza` | REAL | Confianza de la predicción (0–100) |
| `contenedor` | TEXT | Blanco / Negro / Verde |

### Decisiones técnicas

- **`@st.cache_resource`** para el motor de clasificación y la conexión a la base de datos: se inicializan una sola vez por proceso.
- **`st.session_state`** para el historial de la sesión, el estado de la cámara y la huella (hash) de la última imagen procesada, lo que evita clasificaciones duplicadas cuando la interfaz se recarga.
- **Consultas parametrizadas** y acceso sincronizado (`threading.Lock`) a SQLite.
- **Callbacks** (`on_click`, `on_change`) para que los cambios de estado se apliquen antes de redibujar la interfaz.

---

## Requisitos

| Componente | Versión mínima |
|---|---|
| Python | 3.10 |
| Sistema operativo | Windows 10/11, macOS 12+ o Linux |
| Navegador | Chrome, Edge o Firefox (versiones recientes) |
| Hardware | Cámara web integrada o USB |

Dependencias de Python (`requirements.txt`):

| Paquete | Uso |
|---|---|
| `streamlit` ≥ 1.67 | Interfaz web |
| `plotly` | Visualizaciones interactivas |
| `pandas` | Manipulación de datos |
| `numpy` | Cálculo numérico |
| `opencv-python-headless` | Procesamiento de imágenes y ejecución de la red neuronal (`cv2.dnn`) |
| `Pillow` | Lectura de imágenes |

`sqlite3` viene incluido en la biblioteca estándar de Python.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/lassombrita/proyecto-de-quimica.git
cd proyecto-de-quimica
```

### 2. Crear y activar un entorno virtual

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
> Si PowerShell bloquea el script, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. (Opcional) Configurar Visual Studio Code

Instala la extensión **Python** de Microsoft. Luego abre `Ctrl+Shift+P` → **Python: Select Interpreter** y selecciona el intérprete de `venv`.

---

## Uso

### Iniciar la aplicación

```bash
streamlit run app.py
```

La aplicación queda disponible en **http://localhost:8501**. Para usar otro puerto:

```bash
streamlit run app.py --server.port 8502
```

Para detener el servidor, presiona `Ctrl+C` en la terminal.

### Clasificar un residuo

1. En la barra lateral, selecciona **📷 Clasificador en Vivo**.
2. Pulsa **▶️ Encender cámara** y autoriza el acceso a la cámara en el navegador.
3. Ubica el residuo en el centro del encuadre y pulsa **Take Photo**.
4. El panel derecho muestra el material detectado, el contenedor asignado y el nivel de confianza.
5. Pulsa **⏹️ Apagar cámara** al terminar. La cámara también se apaga sola al cambiar de módulo.

### Consultar el tablero analítico

Selecciona **📊 Tablero Estadístico** para ver:

- **Indicadores:** total de residuos, confianza promedio y material más frecuente.
- **Distribución por material:** gráfico de anillo interactivo.
- **Volumen por contenedor:** gráfico de barras.
- **Historial:** las últimas 15 clasificaciones, con opción de ordenar y buscar.

### Reiniciar los datos

El botón **🗑️ Reiniciar datos** de la barra lateral elimina todos los registros de la base de datos y el historial de la sesión.

### Solución de problemas

| Síntoma | Solución |
|---|---|
| `streamlit` no se reconoce como comando | Activa el entorno virtual o usa `python -m streamlit run app.py`. |
| `ModuleNotFoundError: No module named 'cv2'` | Ejecuta `pip install -r requirements.txt` con el entorno activo. |
| La cámara no muestra imagen | Revisa el permiso de cámara del sitio (ícono 🔒 junto a la URL) y cierra otras aplicaciones que la estén usando. |
| La cámara no funciona desde otro equipo de la red | Los navegadores solo permiten la cámara en `localhost` o sobre HTTPS. |
| El puerto 8501 está ocupado | Inicia la aplicación con `--server.port 8502`. |

---

---

## Cómo ejecutar el proyecto

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Abrir la app
```bash
streamlit run app.py
```
Se abre en http://localhost:8501. La base de datos (`data/reciclaje.db`) se crea sola.

### 3. Poblar el tablero con TACO (datos públicos)
Requiere descargar TACO (https://github.com/pedropro/TACO) en una carpeta **fuera** del proyecto.
```bash
# Medir sin escribir en la base (línea base):
python ingesta_publica.py --ruta ../TACO/data --limite 800 --dry-run --reporte-csv docs/resultados/linea_base_taco_v2.csv
# Poblar el tablero (originales + 2 variantes aumentadas por objeto):
python ingesta_publica.py --ruta ../TACO/data --aumentos 2 --distribuir-dias 30
# Deshacer:
python ingesta_publica.py --borrar-origen taco_aug
```
Resultados de la línea base: [RESULTADOS.md](RESULTADOS.md).

### 4. Verificar los recortes (opcional)
```bash
python scripts/diagnostico_recortes.py --ruta ../TACO/data --aumentados
```
Genera `docs/img/diagnostico_recortes.jpg`.

### 5. Reentrenar el modelo
Requiere TrashNet (`dataset-resized.zip` de https://github.com/garythung/trashnet) descomprimido en `entrenamiento/dataset-resized/`.
```bash
pip install -r entrenamiento/requirements-entrenamiento.txt
python entrenamiento/reentrenar.py --taco ../TACO/data
```
Solo reemplaza el modelo si mejora en fotos nunca vistas; guarda el anterior en `models/respaldo/<fecha>/`.

### 6. Restaurar el modelo anterior
```bash
python entrenamiento/reentrenar.py --restaurar models/respaldo/<fecha>
```

### 7. Correr las pruebas automáticas
```bash
python -m pytest -q
```
Usan bases de datos temporales: no tocan `data/reciclaje.db`. Detalle en [docs/VERIFICACION.md](docs/VERIFICACION.md).
Guía para presentar el proyecto: [docs/GUIA_EXPOSICION.md](docs/GUIA_EXPOSICION.md).

### 8. Escáner Edge (`app_main.py`)
```bash
streamlit run app_main.py
```
Usa el modelo real (`vision_edge.ModeloVisionReal`). Sus eventos se guardan con origen `camara_edge`.

---

## Pipeline de 2 etapas

```
cámara ─► Detector "residuo" (YOLO11n, 1 clase) ─► recorte ─► Clasificador (MobileNetV3-Large, 16 clases)
                                                                        │
                                       taxonomia/canecas.yaml (Res. 2184) ◄─┘ ─► Blanca / Negra / Verde
```

Estado: `models/detector_residuo.onnx` y `models/clasificador_material.onnx` existen y la app los usa
por defecto ("Motor 2 Etapas"). La versión actual se entrenó **solo con TACO** y **no pasa** los
criterios de aceptación (`models/reporte_pipeline.txt`): hay que re-entrenar con todas las fuentes.

| Archivo | Rol |
|---|---|
| `datos/descargar.py` | Descarga TACO, TrashNet, ZeroWaste-f y Open Food Facts (Colombia), reanudable |
| `datos/capturar_fotos.py` | Fotos propias con la webcam → `fotos_propias/<sesion>/<clase>/` |
| `datos/construir_manifiesto.py` | Une todas las fuentes en un solo manifiesto (una fila por objeto), sin fugas entre conjuntos |
| `taxonomia/master.yaml` · `canecas.yaml` | 16 clases + alias por fuente · clase → caneca |
| `entrenamiento/pipeline.yaml` · `train_pipeline.py` | Entrena detector y clasificador, calibra, exporta ONNX y evalúa |
| `motor_dos_etapas.py` | Inferencia de producción (onnxruntime / cv2.dnn) usada por `app.py` y `app_main.py` |

```powershell
# 0. Dependencias de entrenamiento (con GPU NVIDIA: instale antes torch con CUDA desde pytorch.org)
pip install -r requirements.txt -r entrenamiento/requirements-entrenamiento.txt
# 1. Descargar los bancos (fuera del repo, en ..\datasets). ZeroWaste pide ~17 GB libres
python datos/descargar.py --todo --borrar-zip
# 2. Fotos propias (opcional pero es lo que más mejora el resultado)
python datos/capturar_fotos.py --sesion cocina_tarde
python datos/capturar_fotos.py --sesion prueba_sala      # examen: nunca se entrena con ella
# 3. Manifiesto
python datos/construir_manifiesto.py --taco ..\datasets\TACO\data --trashnet ..\datasets\trashnet `
    --zerowaste ..\datasets\zerowaste-f --off-productos ..\datasets\off\productos_colombia.jsonl `
    --off-imagenes ..\datasets\off\imagenes --propias fotos_propias
# 4. Pesos base (una vez, con internet) · prueba de humo · entrenamiento completo
python entrenamiento/train_pipeline.py --etapas preparar-pesos
python entrenamiento/train_pipeline.py --rapido
python entrenamiento/train_pipeline.py
# 5. Probar el modelo nuevo en fotos sueltas
python motor_dos_etapas.py foto1.jpg foto2.jpg
```

**Fotos propias:** `fotos_propias/<sesion>/<clase>/*.jpg`. Las sesiones cuyo nombre empieza por
`prueba` se reservan para la evaluación. Los nombres de clase aceptados están en los `alias: propias`
de `taxonomia/master.yaml` (p. ej. `bolsa_blanca`, `servilleta`, `icopor`, `paquete_papas`, `fondo`)
y también el nombre oficial de cada clase (`pet_botella`, `metal_lata`, …).

---

© Johan Lasso · Reciclaje Inteligente
