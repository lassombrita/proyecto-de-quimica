# RESULTADOS - LÍNEA BASE (MobileNetV2 vs Residuos Colombianos)

## Resumen de Ejecución

```
Procesadas: 454/454 en 33.1s (13.7 img/s) | errores: 0
DRY-RUN: se habrían guardado 407 registros.
```

## Métricas del Modelo (Brecha de Dominio)

- **Exactitud global: 14.6%**
- **Exactitud sobre identificados: 15.0%**
- **No identificado (fallback): 3.1%**

## Matriz de Confusión (MobileNetV2 entrenado en TrashNet EE.UU.)

| Clase Real \ Predicha | Metal | No aprovechable | Papel/Cartón | Plástico | Vidrio | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Metal** | 41 | 2 | 3 | 18 | 6 | 1 | **57.7%** |
| **No aprovechable** | 15 | 15 | 7 | 95 | 37 | 2 | **8.8%** |
| **Papel/Cartón** | 3 | 3 | 1 | 69 | 3 | 1 | **0.0%** |
| **Plástico** | 11 | 4 | 1 | 27 | 33 | 4 | **0.0%** |
| **Vidrio** | 3 | 0 | 1 | 9 | 5 | 6 | **25.0%** |

## Análisis de la Brecha de Dominio

### Clases con Mayor Problema:

1. **"Papel/Cartón"**: 0% recall - casi nunca se identifica correctamente
2. **"Plástico"**: 0% recall - también muy mal clasificado
3. **"No aprovechable" (icopor/multicapa)**: solo 8.8% recall - confundido con otras clases

### Clases que Funcionan Mejor:

1. **Metal**: 57.7% recall (mejor desempeño)
2. **Vidrio**: 25.0% recall  
3. **No identificado (fallback)**: 3.1% de las imágenes no fueron clasificadas

### Conclusión:

El modelo MobileNetV2 entrenado con **TrashNet (EE.UU.)** tiene una brecha de dominio severa con los residuos colombianos:
- No reconoce adecuadamente empaques multicapa metalizados, icopor y aerosoles comunes en Colombia
- Confunde frecuentemente las clases entre sí (problemas de sobreajuste al dataset original)

## Siguiente Paso

Ejecutar el aumento de datos (`--aumentos 2 --distribuir-dias 30`) para poblar la base de datos y comparar el rendimiento en condiciones de webcam vs laboratorio.
