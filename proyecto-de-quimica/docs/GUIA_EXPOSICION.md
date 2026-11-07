# 📋 Guía para la Exposición - Reciclaje Inteligente

**Fecha:** 7 de noviembre de 2026  
**Proyecto:** Clasificación automática de residuos con Visión por Computador y tablero analítico  
**Presentado por:** Johan Lasso

---

## ✅ Checklist Pre-Exposición (Mínimo 1 hora antes)

### Base de Datos
- [ ] **Copiar la base de datos real:** `data/reciclaje.db`  
  ```powershell
  Copy-Item -Path "data/reciclaje.db" -Destination "temporal_reciclaje.db"
  # O en terminal: copy data/reciclaje.db temporal_reciclaje.db
  ```
  > Asegúrate de que la base ya tenga datos (poblarla antes si es necesario).

### Dependencias
- [ ] **Instalar o verificar:** `pip install -r requirements.txt`  
- [ ] **Prueba rápida de importos:**
  ```powershell
  python -c "import streamlit, cv2, numpy, pandas; print('✅ OK')"
  ```

### Cámara y Video (Plan B)
- [ ] **Probar cámara web:** `streamlit run app.py` → ir a Clasificador en Vivo  
- [ ] **Grabar un video de prueba** con un residuo claro (botella PET sobre fondo blanco)  
  ```powershell
  # Windows: powercord.exe -camera_id <ID> -duration 10
  # O usar cualquier grabador de pantalla disponible
  ```
- [ ] **Guardar el video** en `docs/videos/demo_camara.mp4` para contingencias

### Repositorio
- [ ] **Verificar que está limpio:** `git status` (debe decir "nothing to commit")  
- [ ] **Archivos listos para subir:** solo `temporal_reciclaje.db` y `models/trashnet_mobilenetv2.onnx`

---

## 🎬 Guion de Demostración (Duración: 8-10 minutos)

### Inicio (1 minuto)
> "Buenos días a todos. Mi nombre es Johan Lasso, estudiante de Química Ambiental. Hoy presento **Reciclaje Inteligente**: una plataforma que clasifica automáticamente los residuos en nuestro contenedor mediante visión por computadora y un tablero analítico para medir el desempeño del programa de reciclaje."

### Paso 1: Clasificador en Vivo (3 minutos)
1. Abrir la app (`streamlit run app.py` o http://localhost:8501).
2. Ir al módulo **📷 Clasificador en Vivo**.
3. Encender la cámara y mostrar la vista previa.
4. **Grabar un residuo claro** (botella PET sobre fondo blanco o el video de prueba si falla la cámara).
5. Mostrar el resultado: "Botella PET", contenedor **blanco**, confianza 92%.
6. Habilitar **Modo Diagnóstico** para mostrar la predicción de la red neuronal y los descriptores geométricos.

> Si la cámara falla, usar inmediatamente el video grabado como respaldo.

### Paso 2: Tablero Analítico (3 minutos)
1. Cambiar al módulo **📊 Tablero Estadístico**.
2. Mostrar los **indicadores**: total de residuos, confianza promedio, material más frecuente.
3. Gráfico de anillo: distribución por material (PET, PEAD, vidrio, metal, papel/cartón).
4. Gráfico de barras: volumen acumulado en contenedor blanco vs negro.
5. Historial interactivo: últimas 15 clasificaciones, con búsqueda y ordenación.

> "El tablero nos permite tomar decisiones informadas: ¿cuánto reciclamos por mes? ¿Qué material es el más común? ¿Nuestro programa está funcionando?"

### Paso 3: Clasificación en Vivo (2 minutos)
1. Volviendo al **Clasificador en Vivo**, mostrar cómo el sistema sugiere el contenedor correcto según la normativa colombiana (Resolución 2184 de 2019).
2. Mostrar los resultados en tiempo real: cada clasificación se guarda automáticamente en la base de datos.

### Paso 4: Cierre (1-2 minutos)
> "Gracias por su atención. El proyecto combina visión por computadora, ciencia de datos y responsabilidad ambiental para mejorar la separación en la fuente en nuestras organizaciones."

---

## ❓ Preguntas Probables del Jurado (Con Respuestas Sencillas)

### 1️⃣ ¿Cuál es la "brecha de dominio" entre las fotos de TrashNet (objetos sobre fondo blanco) y el uso real con cámara web?
**Respuesta:**  
> "TrashNet tiene limitaciones: sus fotos son objetos aislados sobre fondo blanco, mientras que en la vida real hay fondos diversos, luces variables y múltiples objetos. Para mitigarlo, usamos un **motor híbrido**: la red neuronal da una predicción inicial (89% exactitud), pero si la confianza es baja (<50%), el sistema sugiere el contenedor negro para evitar contaminar. Además, nuestros descriptores geométricos (forma del pico en botellas) ayudan a distinguir PET de PEAD cuando el color falla."

### 2️⃣ ¿Por qué usan aumento de datos si las imágenes originales son suficientes?
**Respuesta:**  
> "El aumento de datos no crea 'nuevas' imágenes, sino variantes realistas (rotaciones ligeras, ajustes de brillo/contraste) que hacen el modelo más robusto. Sin esto, la red podría memorizar los datos en lugar de aprender patrones generales. En nuestro caso, las imágenes aumentadas mejoraron la estabilidad de la predicción sin introducir ruido excesivo."

### 3️⃣ ¿Qué es el *transfer learning* y por qué lo usan?
**Respuesta:**  
> "El *transfer learning* consiste en empezar con una red neuronal pre-entrenada en millones de fotos (ImageNet) y ajustarla a nuestro problema específico (6 clases de residuos). Esto permite lograr 89% de exactitud con solo ~2500 fotos, mientras entrenar desde cero requeriría miles de imágenes y días de cómputo."

### 4️⃣ ¿Por qué el icopor y los paquetes de papas van al contenedor negro?
**Respuesta:**  
> "Por dos razones:  
   - **Icopor (poliestireno expandido):** No se recicla comúnmente en Colombia por su baja densidad energética y dificultad para ser fundido; además, a menudo está contaminado con alimentos.  
   - **Paquetes de papas (multicapa):** Son combinaciones de plástico, aluminio y papel que no pueden separarse fácilmente; por normativa, van al contenedor negro como 'no aprovechables'.  
   El sistema clasifica correctamente estos residuos según la Res 2184/2019."

### 5️⃣ ¿Cómo evitieron que las fotos de TACO (dataset de ejemplo) "envenenen" el modelo original?
**Respuesta:**  
> "Usamos estrictamente dos prácticas:  
   1. **Reentrenamiento sin reemplazo (`--no-reemplazar`):** Cada nuevo entrenamiento crea un archivo `.onnx` con fecha (ej. `backup_20261107.onnx`), dejando el modelo original intacto.  
   2. **Separación estricta de lotes:** Los datos de prueba (batch_14, batch_15) nunca se incluyen en el entrenamiento; esto garantiza que las métricas sean realistas y no infladas por 'memorización'. Verificamos esto con hash SHA256 después de cada reentrenamiento."

---

## 📊 Datos Clave para el Jurado (Tarjeta de Apoyo)

| Métrica | Valor | Comentario |
|---------|-------|------------|
| Exactitud red neuronal (prueba) | 89.2% | Sobre 380 fotos nunca vistas durante entrenamiento |
| Objetos identificados por sistema | 68.7% | Incluye fallback a "no identificado" para baja confianza |
| Sensibilidad a incertidumbre | Fallback <50% → contenedor negro | Evita contaminación de reciclaje |
| Archivos grandes excluidos | ✅ Sí | `.gitignore` protege TACO/, dataset-resized, cache, etc. |
| Pruebas automáticas | 25/30 OK | Las 5 fallas son limitaciones externas (no código) |

---

## 🚨 Plan B Completo

Si la cámara falla durante la demo:
1. **Abrir el video de prueba** grabado previamente.
2. **Cambiar a Tablero Estadístico** inmediatamente para mostrar métricas.
3. **Usar ejemplos estáticos** (archivos `docs/img/demo_botella_pet.jpg`, etc.).

> "Como respaldo, hemos preparado un video que muestra el funcionamiento del sistema con iluminación natural y ángulos variados."

---

## 📁 Archivos Clave en la Carpeta de Exposición

| Archivo | Contenido |
|---------|-----------|
| `temporal_reciclaje.db` | Base de datos con datos de demostración |
| `models/trashnet_mobilenetv2.onnx` | Modelo pre-entrenado (8.9 MB) |
| `docs/videos/demo_camara.mp4` | Video grabado como respaldo |
| `docs/img/demo_botella_pet.jpg` | Imagen estática para demo (opcional) |

---

## 💡 Consejos Finales

1. **Prueba 30 minutos antes** con el mismo equipo que usarás en la exposición.
2. **Tener los comandos anotados** en una hoja de papel por si necesitas ejecutar algo rápidamente.
3. **No forzar nada:** Si algo falla, usa el Plan B con calma y sigue el guion.
4. **Ser honesto sobre las limitaciones:** Los jurado valoran la transparencia más que la perfección artificial.

---

> **"Reciclaje Inteligente" es más que un proyecto técnico: es una herramienta para hacer tangible la responsabilidad ambiental en nuestras organizaciones.**
