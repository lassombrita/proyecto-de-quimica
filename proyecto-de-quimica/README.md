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
| PET (resina #1) | Botellas de bebidas | ⚪ Blanco |
| PEAD (resina #2) | Envases de detergente y lácteos | ⚪ Blanco |
| Vidrio | Botellas y frascos | ⚪ Blanco |
| Metal | Latas de aluminio y acero | ⚪ Blanco |
| Papel/Cartón | Cajas, hojas, periódico | ⚪ Blanco |
| No aprovechable | Servilletas, empaques metalizados, residuos contaminados | ⚫ Negro |

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
├── docs/
│   └── AUDITORIA_Y_CORRECCION_VISION.md  # Informe técnico de auditoría y correcciones
├── models/
│   ├── trashnet_mobilenetv2.onnx   # Red neuronal entrenada (se ejecuta con OpenCV)
│   ├── reporte_entrenamiento.txt   # Métricas de exactitud
│   └── licencias/                  # Licencias de los datos y pesos usados
├── entrenamiento/         # Scripts para convertir, entrenar y evaluar el modelo
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

El sistema integra **4 estrategias de clasificación intercambiables en tiempo real** desde la barra lateral:

1. **Motor Básico (Color):** Heurística cromática clásica en espacio HSV central. Demuestra el problema de sesgo cromático (ej. botella PET con jugo café clasificada como cartón).
2. **Motor Geométrico (Forma/Contornos):** Clasificación basada puramente en OpenCV (Canny, descriptores de aspecto, solidez, rectangularidad y análisis de cuello). No depende de color.
3. **Motor IA (MobileNetV2):** Inferencia directa con la red neuronal convolucional entrenada en TrashNet y ejecutada con `cv2.dnn` sobre canales RGB. Sin bloqueos geométricos.
4. **Motor Híbrido Estricto:** Fusión no destructiva donde la red neuronal lidera la decisión del material y la morfología aporta confirmación estructural (+8% confianza) y especialización fina (PET vs PEAD según el centrado del pico).

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

© Johan Lasso · Reciclaje Inteligente
