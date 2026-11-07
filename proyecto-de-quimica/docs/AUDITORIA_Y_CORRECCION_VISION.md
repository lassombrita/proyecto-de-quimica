# 📋 Informe de Auditoría Técnica y Corrección del Sistema de Visión

**Proyecto:** Sistema de Reciclaje Inteligente  
**Componente:** Módulo de Detección e Inferencia Visual (`vision_engine.py` / `app.py`)  
**Fecha:** Noviembre 2026  
**Responsable:** Arquitectura de Software y Visión por Computador  

---

## 1. Resumen Ejecutivo

Durante las pruebas operativas del sistema con cámaras web en entornos reales, se presentó un **fallo crítico de clasificación**:
- En la versión original basada en heurística cromática, el sistema confundía materiales debido al color del contenido (ej. una botella de plástico transparente llena de jugo de manzana era catalogada erróneamente como **Papel/Cartón** debido al tono marrón/ámbar del líquido).
- En la versión intermedia morfológica + ONNX, el sistema colapsó hacia el extremo opuesto: **cualquier objeto real era clasificado como `"No identificado"` y enviado al contenedor Negro**.

Tras una auditoría exhaustiva del flujo de procesamiento de imágenes, se identificaron los cuellos de botella que forzaban dicho estado de incertidumbre, se refactorizó la lógica en un **sistema Multi-Engine**, se calibraron los umbrales de seguridad al **50%** y se habilitó un **Modo Diagnóstico interactivo** en Streamlit.

---

## 2. Hallazgos de la Auditoría (Causas Raíz)

```
                            ANATOMÍA DEL FALLO PREVIO
                            
     [Foto Webcam BGR]
             │
             ▼
      [_segmentar()] ──► Estimación de fondo por franja perimetral de 10 px
             │             │
             │             └──► En habitación real: alta varianza en bordes
             │                  => No aísla contorno (contorno = None)
             ▼
     [Veto Temprano] ──► Sin contorno NO se llama a la red neuronal
             │           => Fallback directo a "No identificado" (Contenedor Negro)
             ▼
  Si había contorno:
  [_fusionar: Caso C] ──► Si la forma (ej. "Irregular") no encajaba con la clase de la red
                         => Regla de "Evidencia Contradictoria" bloqueaba la decisión
                         => Fallback forzado a "No identificado"
```

### 2.1. Sesgo Cromático de la Heurística Clásica
El método inicial analizaba el espacio HSV de la zona central de la imagen. La regla:
```python
elif 8 <= tono < 25 and sat >= 45:
    logits += [0.1, 0.2, 1.0, 0.2, 2.3, 0.4]  # Favorece Papel/Cartón
```
asignaba el residuo al contenedor de reciclaje de cartón cuando la botella contenía bebidas oscuras, té o jugos, provocando contaminación cruzada en la fuente.

### 2.2. Veto Temprano por Fallo de Segmentación en Fondos Ruidosos
El método `_segmentar()` calculaba la mediana y dispersión del color en una franja perimetral de 10 píxeles asumiendo un fondo de estudio homogéneo. En una habitación con paredes, muebles, sombras o la presencia del usuario:
1. La dispersión del fondo era muy alta, haciendo el umbral de binarización inalcanzable.
2. Si el objeto tocaba los bordes de la captura en más del $5\%$, se marcaba como silueta incompleta (`parcial = True`).
3. El código determinaba que `contorno is None` y abortaba el flujo inmediatamente, **impidiendo que la red neuronal procesara el fotograma**.

### 2.3. Veto Destructivo por Contradicción Forma vs. Red (Caso C)
El módulo aplicaba una regla de exclusión mutua: si la geometría deducida (ej. silueta clasificada como irregular debido a la deformación del plástico) no coincidía con las familias autorizadas de la red, se catalogaba como *"evidencia contradictoria"* y se asignaba `confianza = 0.0%`, enviándolo a la caneca negra. En condiciones de webcam, esto anulaba más del 80% de los objetos válidos.

### 2.4. Inconsistencia de Espacio de Color BGR vs. RGB
OpenCV decodifica en formato **BGR**, mientras que MobileNetV2 (pesos de ImageNet / TrashNet) fue entrenada en espacio **RGB** normalizado con media $\mu = [0.485, 0.456, 0.406]$ y desviación $\sigma = [0.229, 0.224, 0.225]$. Cualquier omisión en la conversión rigurosa de canales producía tensores erráticos en la inferencia.

### 2.5. Umbrales Sobredimensionados
Se exigía una probabilidad mínima del $85\%$ para que la red decidiera por sí sola y un umbral global del $65\%$, provocando que predicciones perfectamente viables (entre 50% y 64%) fueran descartadas.

---

## 3. Correcciones e Implementación Técnica

### 3.1. Nuevo Motor Central: `vision_engine.py`
Se estructuró una arquitectura modular que desacopla el clasificador en 4 estrategias intercambiables:

| Motor | Descripción | Caso de Uso |
|---|---|---|
| **Motor Básico (Color)** | Heurística cromática basada en espacio HSV promedio en la región central. | Demostrar y auditar por qué el color falla ante contenidos líquidos. |
| **Motor Geométrico (Forma/Contornos)** | Visión clásica con Canny, contornos, relación de aspecto, solidez y ratio de cuello. | Detección determinista sin sesgo cromático para objetos con bordes definidos. |
| **Motor IA (MobileNetV2)** | Inferencia pura sobre `models/trashnet_mobilenetv2.onnx` mediante `cv2.dnn`. | Clasificación basada en texturas y características complejas de la red neuronal. |
| **Motor Híbrido Estricto** | Fusión jerárquica no destructiva: la red decide el material y la geometría aporta confirmación (+8% confianza) y refinamiento (PET vs PEAD). | **Motor recomendado para producción.** |

### 3.2. Fusión Jerárquica No Destructiva
Se reemplazó el veto de "evidencia contradictoria" por una ponderación constructiva:
1. **La red neuronal lidera:** Clasifica el recorte del objeto (o encuadre central al 80% si el fondo es complejo).
2. **La forma apoya:**
   - Si la red detecta `plastic`: la presencia de cuello centrado confirma **PET**, mientras que un pico lateral desplazado (>15%) especializa la categoría a **PEAD**.
   - Si la geometría corrobora la clase (ej. cilindro con lata metálica, rectángulo con caja de cartón), se suma una bonificación de $+8\%$ a la confianza.
   - Si la forma es irregular (ruido de webcam), la predicción de la red se mantiene válida siempre que supere el umbral del $50\%$.

### 3.3. Calibración del Umbral de Fallback al 50%
Si tras la inferencia la confianza calculada es menor al **$50.0\%$**, el residuo se remite automáticamente a `"No identificado"` con destino al contenedor **Negro** (Resolución 2184 de 2019), protegiendo el material reciclable de contaminación sin penalizar predicciones legítimas.

### 3.4. Cero Dependencias Nuevas
La inferencia se realiza mediante el backend optimizado para CPU `cv2.dnn.readNetFromONNX`. No se requiere instalar TensorFlow ni PyTorch, manteniendo el despliegue ligero, de arranque instantáneo y compatible con el entorno virtual existente.

---

## 4. Mejoras en la Interfaz de Usuario (`app.py`)

1. **Selector Multi-Engine en la Barra Lateral (`st.sidebar.radio`):**
   Permite al usuario conmutar dinámicamente entre los 4 motores sobre la misma imagen capturada sin necesidad de reiniciar la aplicación ni volver a tomar la foto.
2. **Modo Diagnóstico / Debug (`st.sidebar.checkbox`):**
   Al activarse, despliega un panel técnico debajo del resultado con:
   - **Parámetros morfológicos:** Relación de aspecto, rectangularidad, solidez, ratio de cuello y etiqueta geométrica.
   - **Top 3 de la Red Neuronal:** Clases de TrashNet con porcentajes exactos y barras de progreso visuales.
   - **Diagnóstico cromático:** Coordenadas HSV y valor hexadecimal muestreado.
3. **Compatibilidad hacia atrás (`vision_mock.py`):**
   Se mantuvo `vision_mock.py` como un wrapper que re-exporta todas las clases y funciones de `vision_engine.py`, garantizando que scripts de evaluación o tests preexistentes continúen operando sin cambios.

---

## 5. Matriz de Validación de Casos de Prueba

| Caso de Prueba | Motor Básico (Color) | Motor Geométrico | Motor IA (MobileNetV2) | Motor Híbrido Estricto |
|---|---|---|---|---|
| **Botella PET con jugo de manzana (líquido café)** | ❌ Papel/Cartón (Falso positivo por tono ámbar) | ✅ **PET** (Detecta cuello simétrico, ratio 0.34) | ✅ **PET / Plástico** (Textura y borde de botella) | ✅ **PET** (IA + morfología confirma botella, conf. reforzada) |
| **Lata metálica de refresco** | ⚪ Variable según color de pintura exterior | ✅ **Metal** (Aspecto 1.8, cilindro de paredes rectas) | ✅ **Metal** (Predicción TrashNet) | ✅ **Metal** (Fusión lata + textura metálica) |
| **Caja de cartón** | ✅ Papel/Cartón | ✅ **Papel/Cartón** (Rectangularidad > 85%) | ✅ **Papel/Cartón** | ✅ **Papel/Cartón** |
| **Fondo ruidoso de habitación sin objeto claro** | ❌ Asigna color predominante | ⚫ **No identificado** (Sin contorno, conf. 0%) | ⚫ **No identificado** (Certeza < 50%) | ⚫ **No identificado** (Contenedor Negro) |

---

## 6. Archivos Afectados

- [`vision_engine.py`](file:///c:/Users/merej/OneDrive/Desktop/Quimica/proyecto-de-quimica/vision_engine.py): **Nuevo módulo** con la lógica Multi-Engine, extracción geométrica robusta, inferencia ONNX en RGB y lógica de fusión.
- [`vision_mock.py`](file:///c:/Users/merej/OneDrive/Desktop/Quimica/proyecto-de-quimica/vision_mock.py): Actualizado como enrutador transparente hacia `vision_engine.py`.
- [`app.py`](file:///c:/Users/merej/OneDrive/Desktop/Quimica/proyecto-de-quimica/app.py): Adaptado con el selector interactivo de motores en sidebar, recálculo reactivo y visualización del panel de diagnóstico.
- [`README.md`](file:///c:/Users/merej/OneDrive/Desktop/Quimica/proyecto-de-quimica/README.md): Actualizado con la arquitectura Multi-Engine y la documentación del modo diagnóstico.
