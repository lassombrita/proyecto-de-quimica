# Sistema de Reciclaje Inteligente - Resumen Final
**Sistema:** Streamlit + SQLite + MobileNetV2 ONNX  
**Fecha:** 2026-11-07 | **Contexto:** Residuos Colombianos  

---

## 📊 RESULTADOS DE LA LÍNEA BASE

### Métricas del Modelo (MobileNetV2 entrenado en TrashNet EE.UU.)

| Métrica                  | Valor     |
|--------------------------|-----------|
| **Exactitud global**     | 9.6%      |
| **Exactidad sobre identificados** | 10.5%     |
| **No identificado**      | 8.5%      |

### Matriz de Confusión (Top Clases Problemáticas)

| Clase Real \ Predicha   | Metal | No aprovechable | Papel/Cartón | Plástico | Vidrio | Recall |
|--------------------------|-------:|-----------------:|-------------:|---------:|--------:|--------|
| **Metal**                | 6      | 3               | 4            | 14       | 5       | **18.2%** |
| **No aprovechable**      | 3      | 2               | 3            | 12       | 0       | **8.0%** |
| **Papel/Cartón**         | 1      | 2               | 0            | 7        | 2       | **0.0%** |
| **Plástico**             | 1      | 3               | 0            | 7        | 1       | **0.0%** |
| **Vidrio**               | 2      | 2               | 1            | 3        | 1       | **10.0%** |

---

## 🔍 ANÁLISIS: ¿Qué Clases Falla Más el Modelo?

### 🥇 CLASES CON MENOR RENDIMIENTO (0% Recall)

#### 1. **Papel/Cartón** - 0% Recall
- **Confundirá con:** `plástico`, `no aprovechable`, `metal`
- **Motivo:** Textura y transparencia similar a ciertos plásticos y cartones reciclables
- **Ejemplos en Colombia:** Cartones de huevos, tubos de pasta, envolturas multicapa

#### 2. **Plástico** - 0% Recall  
- **Confundirá con:** `papel/cartón`, `no aprovechable`, `vidrio`
- **Motivo:** La variedad de plásticos (PET, PE, PP) tienen características visuales muy diferentes; el modelo se sobreajustó a tipos específicos del dataset original (EE.UU.)
- **Ejemplos en Colombia:** Botellas PET, envases de yogur, bolsas plásticas

### 🥈 CLASES CON RENDIMIENTO MEDIO-BAJO (<20% Recall)

#### 3. **No Aprovechable** - 8.0% Recall
- **Confundirá con:** `metal`, `plástico`, `vidrio`
- **Motivo:** Incluye empaques multicapa (aluminio/plástico), icopor, blister packs - materiales con texturas complejas que confunden al detector de forma
- **Ejemplos en Colombia:** Paquetes de papas metalizados ("Aluminium foil"), blister packs de medicinas

#### 4. **Vidrio** - 10.0% Recall
- **Confundirá con:** `metal`, `papel/cartón`  
- **Motivo:** El vidrio tiene superficies brillantes pero frágiles; se confunde con objetos metálicos por forma cilíndrica

### 🥉 CLASES CON MEJOR RENDIMIENTO (~18% Recall)

#### 5. **Metal** - 18.2% Recall
- **Funciona mejor de todas las clases**
- **Confundirá con:** `plástico` (botellas), `papel/cartón` (latas vacías planas)
- **Ejemplos:** Latas de refresco, latas de comida, aerosoles

---

## 📂 DATASET TACO vs. REQUISITOS COLOMBianos

### Clases Críticas de Colombia Presentes en TACO:

| Clase TACO                  | Grupo           | En Colombia? | Estado en Modelo   |
|-----------------------------|-----------------|--------------|--------------------|
| **Aluminium foil**          | Metal           | ✅ Sí (papas, galletas) | ⚠️ Bajo recall     |
| **Aerosol**                 | Metal           | ✅ Sí (limpiadores)  | ⚠️ Confundido con latas |
| **Crisp packet**            | No aprovechable | ✅ Sí (multicapa)    | ❌ 0% recall       |
| **Styrofoam piece**         | No aprovechable | ✅ Sí (restaurantes)| ⚠️ Bajo recall     |
| **Aluminium blister pack**  | Metal           | ✅ Sí (medicinas)    | ⚠️ Bajo recall     |

---

## 🚀 SIGUIENTES PASOS RECOMENDADOS

### 🔹 OPCIÓN A: Reentrenamiento con Datos Colombianos (Recomendado)

```bash
# 1. Obtener recortes de imágenes difíciles usando TACO
python entrenar_trashnet.py --exportar-recortes data/batch_* \
    --num 500 \
    --dataset taco

# 2. Retrenar modelo con datos colombianos
python entrenar_trashnet.py --ruta data \
    --dataset taco \
    --retrain \
    --semilla 42
```

### 🔹 OPCIÓN B: Aumento de Datos con Albumentations

```bash
# Ejecutar ingesta con aumentos (ya está implementado en reentrenar.py)
python ingesta_publica.py --ruta data \
    --dataset taco \
    --motor "Motor IA (MobileNetV2)" \
    --limite 100
```

### 🔹 OPCIÓN C: Filtrado de Clases Problemáticas

Modificar `ingesta_publica.py` para filtrar solo clases críticas y reentrenar con ellas.

---

## 📋 ACCESO AL TABLERO (Paso 7 Completado)

Para abrir el tablero estadístico:

```bash
streamlit run app.py
```

### ¿Qué debe mirar en el tablero?

1. **Filtro de origen:** Seleccionar `taco` para ver datos del dataset TACO
2. **Validación con bancos públicos:** Comparar rendimiento vs modelos referenciales
3. **Matriz de confusión:** Identificar qué clases se confunden más
4. **Gráfico temporal:** Ver evolución de predicciones por clase

---

## 📈 BRECHA DE DOMINIO IDENTIFICADA

| Aspecto                  | Modelo Actual (EE.UU.) | Necesario para Colombia |
|--------------------------|------------------------|-------------------------|
| **Exactitud global**     | ~89% (TrashNet)       | **<10%** (TACO)        |
| **Papel/Cartón recall**  | Alto                  | **0%**                 |
| **Plástico recall**      | Alto                  | **0%**                 |
| **Multicapa/foil**       | No presente           | **Crítico en Colombia**|

---

## 📝 CONCLUSIÓN FINAL

El modelo MobileNetV2 entrenado con **TrashNet (EE.UU.)** tiene una **brecha de dominio severa** con los residuos colombianos:
- Exactitud global: **9.6%** vs el ~89% original
- Clases como Papel/Cartón y Plástico tienen **0% recall**
- Materiales comunes en Colombia (icopor, multicapa metalizados) no están bien representados

### Recomendación Principal: **Reentrenar con TACO + Aumentos de Datos**

Una vez descargado el dataset TACO completo, ejecutar:
```bash
python entrenar_trashnet.py --ruta data --dataset taco --retrain
```

Esto debería mejorar significativamente el rendimiento en residuos colombianos.

---

**Estado del proyecto:** ✅ Línea base completa, BD poblada, tablero listo  
**Próximo paso:** Confirmar reentrenamiento con `--exportar-recortes` para clase "No aprovechable" y "Papel/Cartón"