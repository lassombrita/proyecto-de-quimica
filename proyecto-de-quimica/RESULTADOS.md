# Resultados — Línea base del modelo frente a TACO (antes de reentrenar)

**Fecha de la medición:** 7 de noviembre de 2026
**Datos:** 800 objetos de TACO (fotos reales de basura), elegidos al azar
**Motor:** Motor Híbrido Estricto (red MobileNetV2 + geometría)
**Detalle por objeto:** [`docs/resultados/linea_base_taco_v1.csv`](docs/resultados/linea_base_taco_v1.csv)

> ⚠️ **Corrección importante.** El primer informe de esta línea base reportaba **12,4 %** de
> exactitud y **0 %** en plástico y papel. Era un error de software: una versión modificada
> de `ingesta_publica.py` había quitado las tildes de las etiquetas reales ("Plastico") y por
> eso nunca coincidían con las predicciones ("Plástico"). Recalculada correctamente, la
> exactitud real es **32,7 %**. El error se corrigió y hoy lo vigila una prueba automática
> (`tests/test_mapeo.py::test_destinos_validos_y_grupos_con_tilde`).

## 1. Resumen

| Indicador | Valor |
|---|---|
| Objetos con etiqueta comparable | 752 de 800 (48 eran orgánicos, peligrosos o sin etiqueta) |
| **Exactitud por material** | **32,7 %** |
| Residuos **no aprovechables** enviados al contenedor correcto (negro) | **17,8 %** (58 de 325) |
| Residuos aprovechables enviados al contenedor correcto (blanco) | 92,0 % (393 de 427) |
| Contenido del contenedor blanco que en realidad es basura | **40,5 %** (267 de 660) |
| Confianza media cuando acierta / cuando se equivoca | 89,5 % / 84,1 % |

## 2. Acierto por material (recall)

| Material real | Objetos | Acierto | Error más frecuente |
|---|---:|---:|---|
| Papel/Cartón | 91 | 81,3 % | Plástico |
| Metal | 92 | 50,0 % | Papel/Cartón |
| Plástico | 210 | 37,6 % | Papel/Cartón |
| Vidrio | 34 | 17,6 % | Papel/Cartón |
| **No aprovechable** | **325** | **12,6 %** | **Papel/Cartón (56 %)** |

### Matriz de confusión (filas = real, columnas = predicho)

| Real \ Predicho | Metal | No aprov. | No identif. | Papel/Cartón | Plástico | Vidrio |
|---|---:|---:|---:|---:|---:|---:|
| Metal | **46** | 4 | 0 | 29 | 10 | 3 |
| No aprovechable | 14 | **41** | 17 | 182 | 67 | 4 |
| Papel/Cartón | 4 | 4 | 1 | **74** | 7 | 1 |
| Plástico | 34 | 13 | 10 | 67 | **79** | 7 |
| Vidrio | 7 | 0 | 2 | 17 | 2 | **6** |

## 3. Clases críticas del consumo colombiano

| Categoría TACO | Objetos | Acierto | El modelo la confunde con |
|---|---:|---:|---|
| Styrofoam piece (icopor) | 22 | 13,6 % | Papel/Cartón |
| Crisp packet (paquete de papas) | 10 | 20,0 % | Papel/Cartón |
| Foam cup (vaso de icopor) | 7 | 0 % | Papel/Cartón |
| Foam food container (coca de icopor) | 2 | 0 % | Papel/Cartón |
| Aerosol | 4 | 75,0 % | — |

Las muestras críticas son pocas (TACO tiene pocos ejemplos de estas clases); los porcentajes
son orientativos. Por eso conviene complementar con fotos propias (`--fotos` en
`entrenamiento/reentrenar.py`).

## 4. Interpretación (para la exposición)

1. **La brecha de dominio es real y tiene una forma concreta:** el modelo aprendió con
   TrashNet (objetos limpios sobre fondo blanco) y ante fotos reales tiende a responder
   «Papel/Cartón» (49 % de todas sus predicciones). El icopor es blanco y opaco, y los
   empaques arrugados se parecen a papel: por eso van al lugar equivocado.
2. **El riesgo ambiental es la contaminación del contenedor blanco:** 4 de cada 10 residuos
   que el sistema mandaría al blanco son en realidad no aprovechables.
3. **El modelo se equivoca con seguridad** (84 % de confianza promedio en los errores):
   el umbral de incertidumbre (50 %) no alcanza a proteger; hay que corregir el modelo,
   no solo el umbral.
4. **Siguiente paso:** reentrenar la capa final con TACO + TrashNet
   (`python entrenamiento/reentrenar.py --taco <TACO/data>`) y comparar con esta línea base.

## 5. Cómo repetir la medición

```bash
python ingesta_publica.py --ruta <ruta de TACO/data> --limite 800 --dry-run --reporte-csv docs/resultados/linea_base_taco_v2.csv
```

Nota: el CSV v1 se generó con una versión anterior del script de ingesta (margen de recorte
15 %, sin variantes aumentadas). La versión actual usa margen de 30 % y además reporta la
variante «webcam»; los números pueden variar unos puntos.

## 6. Resultados del Reentrenamiento con TACO (`reentrenar.py`)

Se ejecutó la prueba de reentrenamiento combinando el dataset **TACO** (con recortes de cajas delimitadoras COCO y aumentos de simulación de webcam) con **TrashNet** (para evitar olvido catastrófico de clases base).

### Comparación en conjunto de PRUEBA (imágenes nunca vistas)

Ejecución con 300 muestras (`--taco TACO/data --limite 300 --no-reemplazar`):

- **Hiperparámetro elegido**: $C = 0.03$
- **Exactitud balanceada en validación**: $67.3\%$
- **Ejemplos procesados (con aumentos)**: 654

| Dataset | Condición | Muestras ($n$) | Exactitud (Actual $\to$ Nuevo) | Recall por grupos (Actual $\to$ Nuevo) |
|---|---|:---:|:---:|:---:|
| **TACO** | Limpia | 14 | **21.4% $\to$ 50.0%** (+$28.6\%$) | **41.7% $\to$ 61.7%** (+$20.0\%$) |
| **TACO** | Webcam | 14 | **28.6% $\to$ 35.7%** (+$7.1\%$) | 45.8% $\to$ 24.2% |
| **TrashNet** | Limpia | 21 | 95.2% $\to$ 71.4% | 97.1% $\to$ 63.3% |
| **TrashNet** | Webcam | 21 | 81.0% $\to$ 66.7% | 65.5% $\to$ 59.3% |

### Categorías críticas
- **Icopor (*Styrofoam piece*):** Pasó de **0.0% a 100.0%** de acierto en prueba.

### Política de salvaguarda de producción
El sistema aplicó la regla de decisión: aunque el recall en TACO aumentó significativamente (demostrando reducción de la brecha de dominio), la exactitud en TrashNet tuvo una caída superior a la tolerancia del 3%. Por lo tanto, el modelo en producción se preservó intacto y el candidato se archivó de forma segura en:
`entrenamiento/salidas/modelo_20261107_012939.onnx` (con reporte detallado en `reporte_20261107_012939.txt`).

---

## 7. Herramientas de Verificación y Diagnóstico Visual

Para auditar y validar los recortes antes de reentrenar o inferir:

```bash
# Generar cuadrícula visual de recortes COCO con etiquetas y variantes de webcam:
python scripts/diagnostico_recortes.py --ruta TACO/data --aumentados

# Ejecutar pruebas unitarias de doctest en reentrenamiento:
python -m doctest entrenamiento/reentrenar.py -v

# Probar reentrenamiento en modo dry-run (sin tocar modelo de producción):
python entrenamiento/reentrenar.py --taco TACO/data --limite 200 --no-reemplazar
```
La cuadrícula generada queda disponible en: `docs/img/diagnostico_recortes.jpg`.

## 6. Después del reentrenamiento (7 de noviembre de 2026)

Capa final reentrenada con TACO (lotes 1-11) + TrashNet, con aumento de datos (C = 0,003).
Evaluado en fotos que el modelo nunca vio (TACO lotes 14-15 y prueba de TrashNet).

| Prueba | Antes | Después |
|---|---:|---:|
| TACO, foto limpia (286 objetos) | 21,3 % | 54,9 % |
| TACO, condiciones de webcam | 21,0 % | 57,0 % |
| Styrofoam piece (icopor, 16) | 6,2 % | 87,5 % |
| Foam cup (vaso de icopor, 10) | 0 % | 50,0 % |
| TrashNet, foto de estudio (380) | 89,2 % | 86,6 % |
| TrashNet, condiciones de webcam | 81,6 % | 83,7 % |

El modelo anterior está respaldado en models/respaldo/20261107_233116.
Reporte completo: docs/resultados/reporte_reentrenamiento_20261107.txt.
Nota: Crisp packet y Foam food container solo tienen 2 fotos de prueba cada uno; no son concluyentes.

